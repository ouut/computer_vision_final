# CV Project — Demo Script & Architecture Overview

---

## Part 1: Live Demo (6 Steps)

### Step 1 — Activate the Environment

```bash
cd cv_project
conda activate cv_project
```

You should see `(cv_project)` in your terminal prompt. All dependencies (PyTorch, NumPy, Pandas) are already installed.

### Step 2 — Launch the Emulator

Open **Eden** (Nintendo Switch emulator). Go to **Settings → Controls → DSU Client** and set:

```
Server: 127.0.0.1
Port:   26760
```

Eden will connect to our DSU server once it starts. You can leave Eden running in the background.

### Step 3 — Start the Interactive Console

```bash
python main.py
```

You will see:

```
Loading model (stgcn_pretrained.pth) ...
  device: cpu  |  6 gestures: Back, Down, Front, Jump, Left, Right
  6/6 gestures mapped to actions

==============================================================
  Motion Play — Gesture-to-Game  Interactive Console
==============================================================
  gestures:     6 (Back, Down, Front, Jump, Left, Right)
  DSU server:   udp://127.0.0.1:26760
  frame-count:  10  (0.17s @ 60Hz)
  dataset:      dataset/
--------------------------------------------------------------
  number       pick a CSV from the list
  list         show the file list again
  /path/to.csv classify a custom file
  q            quit
```

The DSU server is now listening on UDP port 26760. Eden should connect automatically — the prompt will change from `[waiting]` to `[1 connected]`.

### Step 4 — Pick a File and Predict

Type a number from the list (e.g. `36` for a Left gesture CSV):

```
csv [1 connected] > 36
```

The console prints:

```
  file: dataset/Left/Chao_Left_2026-06-10-134423.csv

  prediction: Left   (confidence 21.3%)

  ranking:
    Left      cos=+0.312   21.3%  ######
    Back      cos=+0.287   16.5%  ####
    Right     cos=+0.253   11.7%  ###
    Jump      cos=+0.198    6.7%  ##
    Down      cos=+0.142    3.8%  #
    Front     cos=+0.089    1.9%  #

  ▶ action:    StickL:-1:0:10
    duration:   10 frames (0.17s @ 60Hz)
    sending to  127.0.0.1:54321
```

The model predicted **Left** with 21.3% confidence. The action `StickL:-1:0:10` (tilt left stick left for 10 frames) was sent to Eden.

### Step 5 — Try Another Gesture

Type another number, for example `9` (a Down gesture):

```
csv [1 connected] > 9

  file: dataset/Down/Chao_Down_2026-06-10-134537.csv

  prediction: Down   (confidence 20.8%)

  ranking:
    ...

  ▶ action:    ZL10
    duration:   10 frames (0.17s @ 60Hz)
    sending to  127.0.0.1:54321
```

You can keep picking files — the console loops until you quit.

### Step 6 — Quit

```
csv [1 connected] > q
quit
Shutting down DSU server...
Goodbye.
```

---

## Part 2: Project Architecture

### The Big Picture

```
┌──────────────────────────────────────────────────────────────────┐
│                        CV PROJECT                                 │
│                                                                   │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌─────────┐ │
│  │ ARKit    │    │  ST-GCN      │    │ Gesture  │    │  DSU    │ │
│  │ CSV      │───▶│  Embedding   │───▶│ → Action │───▶│  UDP    │ │
│  │ (91 jts) │    │  (256-dim)   │    │  Mapping │    │  Server │ │
│  └──────────┘    └──────────────┘    └──────────┘    └────┬────┘ │
│                                                           │      │
└───────────────────────────────────────────────────────────┼──────┘
                                                            │
                                                    ┌───────▼──────┐
                                                    │  Eden (Switch│
                                                    │  emulator)   │
                                                    │  Port 26760  │
                                                    └──────────────┘
```

### Pipeline — Step by Step

| # | Stage | What Happens | File |
|---|-------|-------------|------|
| 1 | **Read CSV** | Load ARKit body-tracking data: 91 joints × XYZ positions | `arkit.py` |
| 2 | **Map to COCO12** | Select 12 key joints (shoulders, elbows, wrists, hips, knees, ankles) | `pose_coco12.py` |
| 3 | **Preprocess** | Center hips at origin, scale by torso length, align first-frame facing direction, resample to 300 frames | `pose_coco12.py` |
| 4 | **Embed** | Feed the (3, 300, 12) tensor through the ST-GCN → get a 256-dimension vector (unit length) | `stgcn.py` |
| 5 | **Match** | Dot product with 6 gesture template vectors → cosine similarity scores | `main.py` |
| 6 | **Softmax** | Temperature-scaled softmax (T=10) → pick the highest score as the prediction | `main.py` |
| 7 | **Map to Action** | Look up the predicted gesture in `action_mapping.json` → get an action sequence string (e.g. `ZL10`) | `main.py` |
| 8 | **Parse Sequence** | Convert `ZL10` into 10 frame objects: `[{buttons: ["ZL"]}, ...]` | `sequence.py` |
| 9 | **Send via DSU** | Build 80-byte DSU controller packets, stream at 60 Hz to all connected emulator clients | `udp_controller.py` |


### Key Design Decisions

**Why cosine similarity instead of a classifier?**
The model was trained on dance videos (AIST++ dataset), not on our 6 gestures. We use it as a *feature extractor*: it turns any body motion into a 256-number summary. By comparing new clips to stored templates with cosine similarity, we can add new gestures without retraining — just record a few examples, average their embeddings, and add the result to `gesture_bank.json`.

**Why a DSU server instead of keyboard emulation?**
The DSU (Cemuhook) protocol is the standard way to feed motion-controlled input into Switch emulators. It handles button presses, analog sticks, and motion sensors at 60 Hz. Eden supports it directly — no extra setup needed.

**Why the interactive console design?**
There are 54 CSV recordings in the dataset (6 gestures × 3 people × 3 takes). A command-line loop lets you quickly test different files without restarting the DSU server. The server stays alive between predictions, so the emulator connection is kept warm.

### File Map

```
cv_project/
│
├── main.py                 ◀── entry point: interactive loop + orchestration
├── action_mapping.json     ◀── config: gesture name → button sequence
│
├── stgcn.py                ◀── ML model: ST-GCN definition (from training)
├── pose_coco12.py          ◀── preprocessing: joint mapping + normalization (from training)
├── arkit.py                ◀── data I/O: CSV reader (from training)
├── dataset_utils.py        ◀── helper: CSV → embedding in one call (from training)
│
├── udp_controller.py       ◀── network: DSU protocol server, 60 Hz send loop (from server)
├── sequence.py             ◀── parser: "A10|B5" → per-frame controller state (from server)
│
├── stgcn_pretrained.pth    ◀── weights: trained on 1,408 AIST++ dance clips
├── gesture_bank.json       ◀── data: 6 gesture template vectors (256-D each)
│
├── dataset/                ◀── test data: 54 ARKit CSV recordings
└── README.md               ◀── setup guide + usage instructions
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| Deep Learning | PyTorch (CPU-only inference) |
| Model | ST-GCN (Spatial-Temporal Graph Convolutional Network) |
| Data Format | ARKit CSV (91 joints × XYZ) |
| Network Protocol | DSU / Cemuhook (UDP, port 26760) |
| Concurrency | asyncio (DSU server + interactive input) |
| Environment | conda (isolated, reproducible) |
