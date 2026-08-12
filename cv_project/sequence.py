"""Frame-based action sequence parser.

Format:  {phase}_{phase}_...   where _ separates sequential phases.
         {token}|{token}|...   where | separates parallel (simultaneous) actions.

  Button:  {ButtonName}{FrameCount}     e.g. A4, B16, ZL8, PLUS4
  Stick:   Stick{L|R}:{x}:{y}:{frames}  e.g. StickL:1:-1:10
  D-Pad:   {UP|DOWN|LEFT|RIGHT}{FrameCount}  e.g. UP4, LEFT8

Examples:
  A4_B16          serial:  A 4 frames, then B 16 frames       (20 frames total)
  A4|B4           parallel: A + B together for 4 frames       (4 frames total)
  A4|B16_X4       mixed:   A+B together, A releases at f4,
                           B continues to f15, then X 4 fr.   (20 frames total)
"""

from typing import List, Optional, Tuple

# ── Known button names (sorted longest-first for greedy prefix match) ──
# NOTE: STICK_L / STICK_R contain "_" (the action separator) so use
# L3 / R3 as aliases instead.  These map to left/right stick clicks.
_BUTTON_NAMES = [
    "RIGHT",                        # 5 chars
    "MINUS",                        # 5 chars
    "LEFT", "DOWN", "PLUS",         # 4 chars
    "HOME",                         # 4 chars
    "L3", "R3",                     # 2 chars — stick clicks (→ STICK_L / STICK_R)
    "ZL", "ZR", "UP",               # 2 chars
    "A", "B", "X", "Y",             # 1 char — face buttons
    "L", "R",                       # 1 char — shoulders
]

# Map sequence button names → canonical names for downstream consumers
_BUTTON_CANONICAL = {
    "L3": "STICK_L",
    "R3": "STICK_R",
}


def _match_button(token: str) -> Tuple[Optional[str], str]:
    """Greedy longest-prefix match against known button names.

    Returns (canonical_button_name, frame_str) or (None, "").
    The frame_str must be a non-empty digit string after the name.
    """
    for name in _BUTTON_NAMES:
        if token.startswith(name):
            remainder = token[len(name):]
            if remainder and remainder.isdigit():
                return _BUTTON_CANONICAL.get(name, name), remainder
    return None, ""


def _parse_one_action(token: str) -> Optional[dict]:
    """Parse a single action token into a dict.

    Returns: {"type": "button", "name": str, "frames": int}
         or: {"type": "stick", "side": "left"/"right", "x": float, "y": float, "frames": int}
         or: None on parse failure (warning printed).
    """
    token = token.strip()

    # ── Stick action: StickL:x:y:frames  or  StickR:x:y:frames ──
    if token.startswith("StickL:") or token.startswith("StickR:"):
        try:
            parts = token.split(":")
            side = "left" if parts[0] == "StickL" else "right"
            return {
                "type": "stick",
                "side": side,
                "x": float(parts[1]),
                "y": float(parts[2]),
                "frames": int(parts[3]),
            }
        except (IndexError, ValueError) as exc:
            print(f"[sequence] Bad stick token {token!r}: {exc}")
            return None

    # ── Button / D-Pad action ──
    button_name, frame_str = _match_button(token)
    if button_name is None:
        print(f"[sequence] Unknown button in token {token!r}")
        return None
    try:
        frame_count = int(frame_str)
    except ValueError:
        print(f"[sequence] Invalid frame count in token {token!r}")
        return None
    return {"type": "button", "name": button_name, "frames": frame_count}


def _neutral_frame() -> dict:
    """Return a neutral (no-action) frame dict."""
    return {"buttons": [], "left_stick": (0.0, 0.0), "right_stick": (0.0, 0.0)}


def parse_frame_sequence(seq_str: str) -> List[dict]:
    """Parse a frame-based sequence string into a flat list of per-frame states.

    Each element: {"buttons": [str, ...], "left_stick": (x, y), "right_stick": (x, y)}

    Returns an empty list for empty / whitespace-only input.
    Unknown tokens are skipped with a warning.
    """
    if not seq_str or not seq_str.strip():
        return []

    phases = seq_str.split("_")
    all_frames: List[dict] = []

    for phase in phases:
        phase = phase.strip()
        if not phase:
            continue

        # Split by | for parallel actions within this phase
        parallel_tokens = phase.split("|")
        actions: List[dict] = []
        for tok in parallel_tokens:
            action = _parse_one_action(tok)
            if action is not None:
                actions.append(action)

        if not actions:
            continue

        # Phase duration = longest action in the group
        phase_len = max(a["frames"] for a in actions)

        for fi in range(phase_len):
            frame = _neutral_frame()
            for a in actions:
                if fi >= a["frames"]:
                    continue  # this action has finished

                if a["type"] == "button":
                    frame["buttons"].append(a["name"])
                elif a["type"] == "stick":
                    stick_key = f"{a['side']}_stick"
                    frame[stick_key] = (a["x"], a["y"])

            all_frames.append(frame)

    return all_frames
