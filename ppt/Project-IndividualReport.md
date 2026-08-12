# Project — Final Individual Report

**Name:** Chao Chen
**Team:** Lei Li · Chao Chen · Chenghua Jiang
**Instructor:** Mana Shahriari
**Course:** Computer Vision Project

---

## 1. Contribution

My role in this project was **system architect and integration engineer**. I was responsible for taking the ST-GCN gesture recognition model (built by Lei) and the ARKit gesture dataset (collected by Chenghua) and turning them into a working, interactive demo that controls a real game.

I designed the overall pipeline architecture that connects three stages: CSV file input → gesture prediction → game control output. I chose to combine the two original sub-projects (`capstone_training` for model inference and `capstone_server` for real-time control) into a single standalone application called `cv_project`. This decision eliminated the need for two separate codebases and made the demo portable — anyone can clone one folder, create one conda environment, and run one command.

My most significant technical contribution was implementing the **DSU (Cemuhook) protocol server** (`udp_controller.py`). This is the bridge between computer vision and game input. The DSU protocol is an open UDP-based standard that Nintendo Switch emulators use to receive motion and button data. I implemented the full protocol from scratch in Python using asyncio: a 60 Hz send loop that streams controller state packets to connected emulator clients, protocol message handling for version exchange, controller list discovery, and data streaming. The server builds 80-byte controller payloads with DS4 button bitmasks, analog stick positions (mapped from float [-1,1] to uint8 [0,255]), and CRC32-verified headers. It supports multiple simultaneous clients with per-client packet counters and automatic timeout-based client expiry.

I also built the **action sequence system** (`sequence.py`) that translates human-readable action descriptions like `"StickL:-1:0:10"` or `"A4|B16"` into per-frame controller states. The parser supports button presses, analog stick tilts, D-pad inputs, parallel (simultaneous) actions separated by `|`, and sequential phases separated by `_`. This DSL makes it easy to define and modify gesture-to-game mappings without touching any code — just edit `action_mapping.json`.

Finally, I created the **interactive console** (`main.py`) that ties everything together into a user-friendly demo experience. It lists all available CSV files, lets the user pick one by number, runs the prediction, displays results in the same format as the original `predict.py`, and sends the mapped action to the emulator. The key design challenge here was keeping the DSU server's 60 Hz send loop alive while waiting for user input. I solved this by using `asyncio.to_thread()` to run `input()` in a background thread, keeping the event loop free for UDP I/O. I also added a connection monitor that prints a notification the moment the emulator connects, so the user gets immediate feedback even while the input prompt is waiting.

---

## 2. Method

### System Architecture

The complete pipeline from input to output is shown below:

```
┌─────────────────────────────────────────────────────────────────┐
│                        cv_project/main.py                        │
│                                                                  │
│  ┌───────────┐    ┌──────────────┐    ┌──────────┐              │
│  │ arkit.py  │    │  stgcn.py    │    │sequence.py│             │
│  │ Read CSV  │───▶│  Embed clip  │───▶│Parse seq  │──┐          │
│  │ 91 joints │    │  → 256-D     │    │string→fr. │  │          │
│  └───────────┘    └──────────────┘    └──────────┘  │          │
│        │                 │                           │          │
│  ┌─────▼─────────────────▼─────┐              ┌──────▼────────┐ │
│  │     pose_coco12.py          │              │udp_controller │ │
│  │  COCO12 mapping             │              │  .py          │ │
│  │  Centering + Scaling        │              │  DSU Server   │ │
│  │  Orientation alignment      │              │  60 Hz send   │─┼─→ Eden
│  │  Resample → 300 frames      │              │  UDP :26760   │ │  emulator
│  └─────────────────────────────┘              └───────────────┘ │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Interactive console (asyncio event loop)                    │ │
│  │  - to_thread(input) → non-blocking user input                │ │
│  │  - _monitor_connection() → emulator connect/disconnect notify│ │
│  │  - Print prediction + ranking + action → loop                │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Code Components

**1. DSU Protocol Server (`udp_controller.py`)**

This is the core of my contribution — a full implementation of the Cemuhook DSU protocol. The server:

- Binds a UDP socket on `0.0.0.0:26760`
- Responds to three DSU message types:
  - `0x100000` (version request) → replies with protocol version 1001
  - `0x100001` (controller list) → replies with one DS4 controller in slot 0
  - `0x100002` (data subscription) → registers the client for data streaming
- Runs a 60 Hz send loop that pops one frame from the buffer per tick, builds an 80-byte DS4 controller payload, and sends it to every registered client
- Handles client registration with handshake verification — only addresses that have completed the version/list handshake can subscribe to data, preventing ghost connections from random UDP traffic
- Tracks per-client packet counters and automatically expires clients that stop re-subscribing for 5 seconds

The controller payload builder (`_build_controller_data`) translates button names into DS4 bitmasks:
- Byte 16: Share, L3, R3, Options, D-Up, D-Right, D-Down, D-Left
- Byte 17: ZL, ZR, L, R, X, A, B, Y
- Byte 18: HOME
- Bytes 20-23: Left stick X/Y, Right stick X/Y (uint8, neutral=128)

Analog sticks are mapped from float [-1.0, 1.0] to uint8 [0, 255] using `(value + 1.0) / 2.0 * 255.0`, with DSU Y-axis convention (0=down, 255=up) matching game stick convention.

I recently added a handshake tracking mechanism (`_handshake_seen` set) that prevents the server from registering ghost clients. Only addresses that first send a version request (`0x100000`) or list request (`0x100001`) are allowed to subscribe for data. This fixed a false-positive connection bug where random UDP packets could trigger client registration.

**2. Action Sequence Parser (`sequence.py`)**

This module provides a domain-specific language for describing game controller actions:

```
Format:  {phase}_{phase}_...     where _ separates sequential phases
         {token}|{token}|...     where | separates simultaneous actions

Tokens:
  Button:  {Name}{Frames}           e.g. A10, B4, ZL20
  Stick:   Stick{L|R}:{x}:{y}:{n}   e.g. StickL:-1:0:10
  D-Pad:   {UP|DOWN|LEFT|RIGHT}{n}  e.g. UP4

Examples:
  A4_B16         → A pressed 4 frames, then B pressed 16 frames
  A4|B4          → A+B together for 4 frames
  StickL:1:0:10  → Left stick full right for 10 frames
```

The parser uses greedy longest-prefix button name matching (e.g., `RIGHT` matches before `R`) and returns a flat list of per-frame state dictionaries: `{"buttons": [...], "left_stick": (x, y), "right_stick": (x, y)}`. This list is what the DSU server's send loop consumes frame by frame at 60 Hz.

**3. Gesture-to-Action Mapping (`action_mapping.json`)**

I designed a simple JSON configuration format that maps each recognized gesture to an action sequence string:

```json
{
    "Back":  "StickL:0:-1:{n}",
    "Down":  "ZL{n}",
    "Left":  "StickL:-1:0:{n}",
    "Right": "StickL:1:0:{n}",
    "Front": "StickL:0:1:{n}",
    "Jump":  "A{n}"
}
```

The `{n}` placeholder is replaced at runtime with the `--frame-count` parameter (default 20 frames = 0.33 seconds at 60 Hz). This design separates gesture semantics from game mechanics — changing what button a gesture triggers requires editing only this JSON file, not the Python code.

**4. Interactive Console (`main.py`)**

The console orchestrates the full pipeline with asyncio concurrency:

```
async def main():
    model, gestures, templates = load_predictor(...)   # sync: load ML model
    mapping = load_action_mapping(...)                  # sync: load config
    dsu = DSUServer(...)                               # create UDP server
    await dsu.start()                                  # async: bind + start send loop
    await run_interactive(...)                         # async: user input loop
    await dsu.stop()                                   # async: clean shutdown
```

The key concurrency design: `input()` runs in a thread pool via `asyncio.to_thread()`, so the event loop stays free to process UDP datagrams from the emulator and run the 60 Hz send loop. A background `_monitor_connection()` task polls the client list every 0.5 seconds and prints a message the instant the emulator connects or disconnects.

### Visual: Console Output Flow

```
$ python main.py

Loading model (stgcn_pretrained.pth) ...
  device: cpu  |  6 gestures: Back, Down, Front, Jump, Left, Right
  6/6 gestures mapped to actions
[dsu] listening on 0.0.0.0:26760  server_id=0x87E51E79

==============================================================
  MotionPlay — Gesture-to-Game  Interactive Console
==============================================================
  gestures:     6 (Back, Down, Front, Jump, Left, Right)
  DSU server:   udp://0.0.0.0:26760
  frame-count:  20  (0.33s @ 60Hz)
  dataset:      dataset/
--------------------------------------------------------------
  number       pick a CSV from the list
  list         show the file list again
  /path/to.csv classify a custom file
  q            quit

csv [waiting] >
  ✓ Emulator connected from 192.168.1.5:54321

csv [1 connected] > 36

  file: dataset/Left/Chao_Left_2026-06-10-134423.csv

  prediction: Left   (confidence 21.3%)

  ranking:
    Left      cos=+0.312   21.3%  ######
    Back      cos=+0.287   16.5%  ####
    Right     cos=+0.253   11.7%  ###
    Jump      cos=+0.198    6.7%  ##
    Down      cos=+0.142    3.8%  #
    Front     cos=+0.089    1.9%  #

  ▶ action:    StickL:-1:0:20
    duration:   20 frames (0.33s @ 60Hz)
    sending to  192.168.1.5:54321
```

---

## 3. Self-Assessment

**What I did well:** I successfully integrated three complex subsystems — a deep learning inference pipeline, a real-time UDP network protocol, and an interactive command-line interface — into one cohesive application that runs reliably. The DSU protocol server was particularly challenging because I had to implement the protocol from its specification document with no existing Python reference. Getting the byte-level packet format correct (DS4 button bitmasks, stick value encoding, CRC32 headers) and debugging it against a real emulator required careful attention to detail. I am also proud of the interactive console design — the `asyncio.to_thread()` approach for non-blocking input is elegant and keeps the controller streaming at full 60 Hz even while waiting for user commands.

**Challenges I faced:** The biggest challenge was a subtle bug where the DSU server would falsely register a "ghost client" immediately after startup. The original code accepted any `0x100002` (data subscription) message from any address that sent a properly-formatted DSU packet. Random UDP traffic on the local network occasionally triggered this, causing the server to start streaming controller data to an address that was not really an emulator. I fixed this by adding handshake verification — the server now tracks which addresses have completed the version request or controller list request steps of the DSU protocol, and only allows those addresses to subscribe for data streaming. This taught me the importance of stateful protocol validation in network servers.

**Skills and knowledge gained:** I developed a deep understanding of the DSU/Cemuhook protocol and UDP-based real-time streaming. I learned how to design interactive asyncio applications that balance blocking I/O (user input) with real-time constraints (60 Hz data streaming). I also gained practical experience with the complete lifecycle of a computer vision project — not just training a model and reporting accuracy, but building the infrastructure that turns model output into a real-world application.

**What I would improve:** If I were to do this project again, I would add automated frame capture so the system works with live ARKit streaming instead of pre-recorded CSV files. The current interactive console requires the user to type a file number; a live version would continuously process incoming skeleton frames, run the gesture classifier on a sliding window, and trigger game actions the moment a gesture is detected. I would also add a visual feedback overlay — showing the predicted gesture and confidence on top of the game screen — to make the demo more engaging for presentations.

---

## 4. References

[1] "Cemuhook DSU Protocol Reference," v1993, 2020. [Online]. Available: https://v1993.github.io/cemuhook-protocol/

[2] S. Yan, Y. Xiong, and D. Lin, "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 32, no. 1, 2018.

[3] Apple Inc., "ARKit — ARSkeletonDefinition," *Apple Developer Documentation*, 2024. [Online]. Available: https://developer.apple.com/documentation/arkit/arskeletondefinition

[4] Python Software Foundation, "asyncio — Asynchronous I/O," *Python 3.10 Documentation*, 2023. [Online]. Available: https://docs.python.org/3.10/library/asyncio.html

---

## Declaration

I, **Chao Chen**, declare that the attached project is entirely my own work, completed in accordance with the Seneca Academic Policy. I have not copied or reproduced any part of this project, either manually or electronically, from any unauthorized source. I have not shared my work with others, nor have I received unauthorized assistance in completing this project.
