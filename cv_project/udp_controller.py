"""DSU (Cemuhook) protocol server for Eden emulator control.

Implements the full DSU server protocol:
  - 0x100000  Protocol version exchange
  - 0x100001  Connected controller list
  - 0x100002  Controller data streaming (buttons, sticks, motion)

Acts as a DSU *server* — Eden (the client) subscribes, and we stream
controller state at 60 Hz.  All button / stick state is driven by an
internal frame buffer that can be atomically replaced via push_sequence()
or push_frames().

Packet reference: https://v1993.github.io/cemuhook-protocol/
"""

import asyncio
import struct
import time
import zlib
from typing import Optional, Tuple

from sequence import parse_frame_sequence

# ═══════════════════════════════════════════════════════════════
# DSU protocol constants
# ═══════════════════════════════════════════════════════════════

DSU_MAGIC = b"DSUS"
DSU_VERSION = 1001
DSU_PORT = 26760

MSG_VERSION = 0x100000
MSG_LIST = 0x100001
MSG_DATA = 0x100002

# Client timeout: stop sending if no re-subscription within 5 s
CLIENT_TIMEOUT = 5.0

# ── DS4 button bitmask layout ──────────────────────────────────
# Byte 16 (buttons1): Share, L3, R3, Options, D-Up, D-Right, D-Down, D-Left
#   bit 0 → MINUS (Share)
#   bit 1 → STICK_L (L3)
#   bit 2 → STICK_R (R3)
#   bit 3 → PLUS (Options)
#   bit 4 → UP
#   bit 5 → RIGHT
#   bit 6 → DOWN
#   bit 7 → LEFT
_BYTE16_MAP: dict[str, int] = {
    "MINUS":  1 << 0,
    "STICK_L": 1 << 1,
    "STICK_R": 1 << 2,
    "PLUS":   1 << 3,
    "UP":     1 << 4,
    "RIGHT":  1 << 5,
    "DOWN":   1 << 6,
    "LEFT":   1 << 7,
}

# Byte 17 (buttons2): L2, R2, L1, R1, X, A, B, Y
#   bit 0 → ZL (L2)
#   bit 1 → ZR (R2)
#   bit 2 → L (L1)
#   bit 3 → R (R1)
#   bit 4 → X
#   bit 5 → A
#   bit 6 → B
#   bit 7 → Y
_BYTE17_MAP: dict[str, int] = {
    "ZL":  1 << 0,
    "ZR":  1 << 1,
    "L":    1 << 2,
    "R":    1 << 3,
    "X":    1 << 4,
    "A":    1 << 5,
    "B":    1 << 6,
    "Y":    1 << 7,
}

NEUTRAL_STICK = 128   # uint8 neutral for DS4 (0–255)


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _map_stick(value: float) -> int:
    """Map float [-1.0, 1.0] → uint8 [0, 255], neutral = 0.0 → 128.

    Uses (value + 1) / 2 * 255 for symmetric mapping across the full range.
    DSU Y axis: 0 = down, 255 = up — same convention as game stick y.
    """
    clamped = max(-1.0, min(1.0, value))
    raw = int(round((clamped + 1.0) / 2.0 * 255.0))
    return max(0, min(255, raw))


def _crc32(data: bytes) -> int:
    """CRC32 of *data* (unsigned 32-bit)."""
    return zlib.crc32(data) & 0xFFFFFFFF


# ═══════════════════════════════════════════════════════════════
# DSUServer
# ═══════════════════════════════════════════════════════════════

class DSUServer:
    """Full DSU protocol server.  Runs a 60 Hz send loop that streams
    controller state to registered emulator clients.

    Usage::

        dsu = DSUServer(bind_host="127.0.0.1", port=26760)
        await dsu.start()
        # ...
        dsu.push_sequence("A4_B16_StickL:1:-1:10")
        # ...
        await dsu.stop()
    """

    def __init__(self, bind_host: str = "127.0.0.1", port: int = DSU_PORT):
        self.bind_host = bind_host
        self.port = port
        self._server_id = int(time.monotonic() * 1_000_000) & 0xFFFFFFFF

        # Frame buffer: consumed one per send tick
        self._buffer: list[dict] = []
        self._buffer_pos = 0
        self._buffer_lock = asyncio.Lock()

        # Registered clients:  (host, port) → last_activity
        self._clients: dict[Tuple[str, int], float] = {}

        # Addresses that completed the version/list handshake
        self._handshake_seen: set[Tuple[str, int]] = set()

        # Per-client packet counter
        self._packet_counters: dict[Tuple[str, int], int] = {}

        # Last sent frame (for dashboard key-state display)
        self.last_sent_frame: dict = self._neutral_frame()
        self._last_send_time: float = 0.0  # for FPS tracking

        # Transport
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[asyncio.DatagramProtocol] = None
        self._send_task: Optional[asyncio.Task] = None
        self._running = False

    # ── lifecycle ───────────────────────────────────────────────

    async def start(self) -> "DSUServer":
        """Bind UDP socket and start the 60 Hz send loop."""
        loop = asyncio.get_running_loop()
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: _DSUProtocol(self),
            local_addr=(self.bind_host, self.port),
        )
        self._running = True
        self._send_task = asyncio.create_task(self._send_loop())
        print(f"[dsu] listening on {self.bind_host}:{self.port}"
              f"  server_id=0x{self._server_id:08X}")
        return self

    async def stop(self) -> None:
        """Stop send loop and close socket."""
        self._running = False
        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
        if self._transport:
            self._transport.close()
        print("[dsu] stopped")

    # ── frame buffer ────────────────────────────────────────────

    def push_sequence(self, seq_str: str) -> None:
        """Parse *seq_str* and atomically replace the frame buffer.

        Unknown / malformed tokens are skipped (warnings printed).
        An empty or whitespace-only string is ignored.
        """
        frames = parse_frame_sequence(seq_str)
        if frames:
            asyncio.ensure_future(self._replace_buffer(frames))

    def push_frames(self, frames: list[dict]) -> None:
        """Atomically replace the frame buffer with pre-built *frames*.

        Each element: {"buttons": [...], "left_stick": (x,y), "right_stick": (x,y)}
        """
        if frames:
            asyncio.ensure_future(self._replace_buffer(frames))

    async def _replace_buffer(self, frames: list[dict]) -> None:
        """Coroutine-safe buffer replacement."""
        async with self._buffer_lock:
            self._buffer = frames
            self._buffer_pos = 0

    # ── neutral state ───────────────────────────────────────────

    @staticmethod
    def _neutral_frame() -> dict:
        return {
            "buttons": [],
            "left_stick": (0.0, 0.0),
            "right_stick": (0.0, 0.0),
        }

    # ── protocol dispatch ───────────────────────────────────────

    def handle_datagram(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Dispatch an incoming DSU packet."""
        if len(data) < 20:
            return  # too short for a valid DSU header

        try:
            magic, version, payload_len, _crc, _sid, msg_type = struct.unpack(
                "<4sHHI4sI", data[:20])
        except struct.error:
            return

        if magic not in (b"DSUS", b"DSUC"):
            return

        # Verify CRC
        expected_crc = _crc32(data[:8] + b"\x00\x00\x00\x00" + data[12:])
        if _crc != expected_crc:
            # Accept anyway — CRC mismatch can happen with buggy clients
            pass

        if msg_type == MSG_VERSION:
            self._handshake_seen.add(addr)
            self._on_version(addr)
        elif msg_type == MSG_LIST:
            self._handshake_seen.add(addr)
            self._on_list(addr, data[20:])
        elif msg_type == MSG_DATA:
            if addr in self._handshake_seen:
                self._on_data(addr)

    def _respond(self, addr: Tuple[str, int], msg_type: int, payload: bytes) -> None:
        """Send a DSU response packet to *addr*."""
        header = self._build_header(msg_type, payload)
        try:
            self._transport.sendto(header + payload, addr)
        except OSError:
            pass

    # ── message handlers ────────────────────────────────────────

    def _on_version(self, addr: Tuple[str, int]) -> None:
        """0x100000 — respond with max supported version."""
        payload = struct.pack("<H", DSU_VERSION)
        self._respond(addr, MSG_VERSION, payload)

    def _on_list(self, addr: Tuple[str, int], body: bytes) -> None:
        """0x100001 — respond with one controller in slot 0."""
        # 11-byte shared header + 1 byte is_pad_active=1
        payload = self._build_shared_header(slot=0, connected=True) + b"\x01"
        self._respond(addr, MSG_LIST, payload)

    def _on_data(self, addr: Tuple[str, int]) -> None:
        """0x100002 — register client (re-subscription)."""
        now = time.monotonic()
        is_new = addr not in self._clients
        self._clients[addr] = now
        if addr not in self._packet_counters:
            self._packet_counters[addr] = 0
        if is_new:
            print(f"[dsu] client connected  {addr[0]}:{addr[1]}")

    # ── send loop ───────────────────────────────────────────────

    async def _send_loop(self) -> None:
        """60 Hz loop: pop next frame → build DSU payload → send to all clients."""
        interval = 1.0 / 60.0
        while self._running:
            t0 = time.monotonic()

            # Expire stale clients
            self._expire_clients()

            # Pop next frame (or neutral) — always record for dashboard
            frame = await self._pop_frame()
            self.last_sent_frame = frame
            self._last_send_time = time.monotonic()  # for FPS tracking

            if self._clients:

                # Build 80-byte controller data payload
                payload = self._build_controller_data(frame)

                # Send to every registered client
                dead: list[Tuple[str, int]] = []
                for addr in self._clients:
                    pkt_counter = self._packet_counters.get(addr, 0)
                    # Inject per-client packet counter into payload
                    pkt_payload = (
                        payload[:12] +
                        struct.pack("<I", pkt_counter) +
                        payload[16:]
                    )
                    header = self._build_header(MSG_DATA, pkt_payload)
                    try:
                        self._transport.sendto(header + pkt_payload, addr)
                    except OSError:
                        dead.append(addr)
                    self._packet_counters[addr] = (pkt_counter + 1) & 0xFFFFFFFF

                for addr in dead:
                    self._clients.pop(addr, None)
                    self._packet_counters.pop(addr, None)

            # Maintain ~60 Hz cadence
            elapsed = time.monotonic() - t0
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def _pop_frame(self) -> dict:
        """Pop the next frame from the buffer (or return neutral if empty)."""
        async with self._buffer_lock:
            if self._buffer_pos < len(self._buffer):
                frame = self._buffer[self._buffer_pos]
                self._buffer_pos += 1
                return frame
        return self._neutral_frame()

    def _expire_clients(self) -> None:
        """Remove clients that haven't re-subscribed within CLIENT_TIMEOUT."""
        now = time.monotonic()
        stale = [
            addr for addr, last in self._clients.items()
            if now - last > CLIENT_TIMEOUT
        ]
        for addr in stale:
            self._clients.pop(addr, None)
            self._packet_counters.pop(addr, None)

    # ── packet builders ─────────────────────────────────────────

    def _build_header(self, msg_type: int, payload: bytes) -> bytes:
        """Build the 20-byte DSU header.

        Layout:
            0-3    magic    "DSUS"
            4-5    version  1001 (uint16 LE)
            6-7    length   payload length (uint16 LE) — excludes these 16 bytes
            8-11   crc32    CRC32 of full packet with this field zeroed
            12-15  id       server ID (uint32 LE)
            16-19  type     message type (uint32 LE)
        """
        # msg_type counts toward payload length
        data_len = 4 + len(payload)
        header_no_crc = struct.pack(
            "<4sHHI", DSU_MAGIC, DSU_VERSION, data_len,
            0,  # CRC placeholder
        ) + struct.pack("<I", self._server_id)
        type_bytes = struct.pack("<I", msg_type)
        # Compute CRC over header_no_crc + type + payload (CRC field = 0)
        full = header_no_crc + type_bytes + payload
        crc = _crc32(full)
        # Rebuild header with correct CRC
        return struct.pack(
            "<4sHHI", DSU_MAGIC, DSU_VERSION, data_len, crc,
        ) + struct.pack("<I", self._server_id) + type_bytes

    @staticmethod
    def _build_shared_header(
        slot: int = 0,
        connected: bool = True,
        model: int = 2,          # 2 = full gyro
        conn_type: int = 1,      # 1 = USB
        mac: bytes = b"\x00" * 6,
        battery: int = 0x05,     # full
    ) -> bytes:
        """Build the 11-byte shared response header."""
        state = 2 if connected else 0
        return struct.pack("<BBB", slot, state, model) + struct.pack("<B", conn_type) + mac + struct.pack("<B", battery)

    def _build_controller_data(self, frame: dict) -> bytes:
        """Build the 80-byte controller data payload (0x100002).

        *frame* dict keys:
            buttons:     list of button name strings
            left_stick:  (x: float, y: float)
            right_stick: (x: float, y: float)
        """
        buttons = set(frame.get("buttons", []))
        lx, ly = frame.get("left_stick", (0.0, 0.0))
        rx, ry = frame.get("right_stick", (0.0, 0.0))

        # ── Button bitmasks ──
        byte16 = 0
        byte17 = 0
        home = 0

        for btn in buttons:
            if btn in _BYTE16_MAP:
                byte16 |= _BYTE16_MAP[btn]
            elif btn in _BYTE17_MAP:
                byte17 |= _BYTE17_MAP[btn]
            elif btn == "HOME":
                home = 1
            # Unknown buttons silently skipped

        # ── Sticks ──
        left_x = _map_stick(lx)
        left_y = _map_stick(ly)    # DSU Y: 0=down, 255=up — matches game convention
        right_x = _map_stick(rx)
        right_y = _map_stick(ry)

        # ── Payload assembly ──
        # Shared header (11) + connected (1) + packet_counter placeholder (4)
        # = 16 bytes, then the rest
        shared = self._build_shared_header()
        connected = b"\x01"
        pkt_counter = b"\x00\x00\x00\x00"  # placeholder, filled per-client
        prefix = shared + connected + pkt_counter  # 16 bytes

        # Digital buttons + HOME + touch (bytes 16–19 in payload → offsets 0–3 here)
        btn_bytes = struct.pack("<BBBB", byte16, byte17, home, 0)

        # Sticks (bytes 20–23 in payload → offsets 4–7 here)
        stick_bytes = struct.pack("<BBBB", left_x, left_y, right_x, right_y)

        # Analog button pressure — all 0 for now (12 bytes)
        analog = b"\x00" * 12

        # Touch data — all 0 (12 bytes)
        touch = b"\x00" * 12

        # Motion timestamp (8 bytes) + accel (12) + gyro (12) — all 0
        motion = b"\x00" * 32

        return prefix + btn_bytes + stick_bytes + analog + touch + motion


# ═══════════════════════════════════════════════════════════════
# DSU protocol helper
# ═══════════════════════════════════════════════════════════════

class _DSUProtocol(asyncio.DatagramProtocol):
    """Thin callback shim — forwards datagrams to DSUServer."""

    def __init__(self, server: DSUServer):
        self._server = server

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        self._server.handle_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        print(f"[dsu] transport error: {exc}")
