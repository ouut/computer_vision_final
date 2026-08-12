# CV Project — Gesture-to-Game Controller

Turn body movements into game controls. Record a gesture with your iPhone (ARKit body tracking), and this tool will figure out what gesture it is, then send the matching button press to a Nintendo Switch emulator over UDP.

## How It Works

```
ARKit CSV file              ST-GCN model             Game emulator
(91 body joints)  ──→  gesture prediction  ──→  button press via UDP
                      (Left/Right/Down/...)      (DSU protocol, 60 Hz)
```

1. You drop a CSV recording of a body gesture.
2. An ST-GCN neural network turns the skeleton motion into a 256-number embedding.
3. The embedding is compared against 6 gesture templates (cosine similarity).
4. The best match is mapped to a game action — like pressing **A** or tilting the **left stick**.
5. The action is sent to the Eden emulator through a DSU server at 60 frames per second.

## Project Layout

```
cv_project/
├── main.py              ← Interactive console (start here)
├── action_mapping.json  ← Gesture → button mapping
├── stgcn_pretrained.pth ← Pre-trained model weights
├── gesture_bank.json    ← Gesture template vectors (6 gestures)
├── dataset/             ← Test CSV recordings (54 files)
│   ├── Back/  Down/  Front/  Jump/  Left/  Right/
├── stgcn.py             ← ST-GCN model definition
├── pose_coco12.py       ← Skeleton preprocessing
├── udp_controller.py    ← DSU protocol server
├── sequence.py          ← Action sequence parser
└── requirements.txt     ← Python dependencies
```

## Requirements

- **Python 3.10+** (conda environment recommended)
- **PyTorch** (CPU is fine — no GPU needed for inference)
- **Eden emulator** (or any emulator that supports the DSU/Cemuhook protocol)

## Setup

### 1. Create the conda environment

```bash
cd cv_project
conda create -n cv_project python=3.10 -y
conda activate cv_project
pip install -r requirements.txt
```

### 2. Check that everything works

```bash
python -c "from main import load_predictor; print('OK')"
```

## Usage

### Step 1: Launch Eden and configure DSU

Open your Eden emulator, go to **Settings → Controls → DSU Client**, and set the server address to `127.0.0.1:26760`.

### Step 2: Start the interactive console

```bash
conda activate cv_project
python main.py
```

You will see:

```
==============================================================
  MotionPlay — Gesture-to-Game  Interactive Console
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

### Step 3: Pick a CSV file and predict

Type a number from the list (for example `0`), or type a full path to any CSV file:

```
csv [waiting] > 0

  file: dataset/Back/Chao_Back_2026-06-10-134338.csv

  prediction: Back   (confidence 21.2%)

  ranking:
    Back      cos=+0.312   21.2%  ######
    Left      cos=+0.287   16.5%  ####
    Right     cos=+0.253   11.7%  ###
    Jump      cos=+0.198    6.7%  ##
    Down      cos=+0.142    3.8%  #
    Front     cos=+0.089    1.9%  #

  ▶ action:    StickL:0:-1:10
    duration:   10 frames (0.17s @ 60Hz)
    sending to  127.0.0.1:54321
```

### Step 4: Repeat or quit

You can pick another file right away. Type `q` to exit.

## Gesture → Action Mapping

| Gesture | Action            | What it does in-game |
|---------|-------------------|----------------------|
| Back    | `StickL:0:-1:10`  | Tilt left stick down |
| Down    | `ZL10`            | Press ZL button      |
| Left    | `StickL:-1:0:10`  | Tilt left stick left |
| Right   | `StickL:1:0:10`   | Tilt left stick right|
| Front   | `StickL:0:1:10`   | Tilt left stick up   |
| Jump    | `A10`             | Press A button       |

Edit `action_mapping.json` to change the mapping. The `{n}` placeholder is replaced by `--frame-count` (default: 10).

## Command-Line Options

| Flag | Default | What it does |
|------|---------|--------------|
| `--weights` | `stgcn_pretrained.pth` | Path to model weights |
| `--bank` | `gesture_bank.json` | Path to gesture template bank |
| `--mapping` | `action_mapping.json` | Path to action mapping config |
| `--game-host` | `127.0.0.1` | DSU server bind address |
| `--game-port` | `26760` | DSU server port |
| `--frame-count` | `10` | Frames per action (1 frame = 1/60 s) |
| `--dataset` | `dataset` | Folder with CSV recordings |

Example with custom settings:

```bash
python main.py --frame-count 15 --game-port 26761
```

## Available Gestures

The 6 gestures in the dataset come from 3 people doing 3 takes each:

| Gesture | Body movement        |
|---------|---------------------|
| Back    | Step backward       |
| Down    | Squat down          |
| Front   | Step forward        |
| Jump    | Jump up             |
| Left    | Step to the left    |
| Right   | Step to the right   |

### Known Issue

**Jump** and **Front** gestures have low recognition accuracy (near 0% across users). This is a known limitation — the iPhone's ARKit tracking reports positions relative to the hip, so vertical movement (jumping) and forward movement (stepping) can look almost identical to standing still. The project documentation discusses this in detail.

The four remaining gestures (Back, Down, Left, Right) reach about **73% accuracy** across different users.

## Adding a Custom CSV

If you have your own ARKit recording in CSV format, just drop it anywhere and type the path:

```
csv [waiting] > /Users/me/recordings/my_jump.csv
```

The CSV must have columns: `joint`, `pos_x`, `pos_y`, `pos_z`. Joint names must match Apple's `ARSkeletonDefinition.defaultBody3D` names.
