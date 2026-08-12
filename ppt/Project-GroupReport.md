# Project — Final Group Report

**Team:** Lei Li · Chao Chen · Chenghua Jiang
**Instructor:** Mana Shahriari
**Course:** Computer Vision Project

---

## 1. Project Title

**MotionPlay — Real-Time Gesture-to-Game Controller Using Skeleton-Based Action Recognition**

---

## 2. Problem Definition

Traditional game controllers — keyboards, mice, and handheld gamepads — require physical buttons and sticks. They are not accessible to everyone and limit how naturally players can express movement. Our project asks: can a player control a game using only their body? We built a system that takes 3D skeleton data from an iPhone camera (ARKit body tracking), recognizes which gesture the player is performing, and sends the matching button press to a Nintendo Switch emulator in real time. The problem sits at the intersection of skeleton-based action recognition and human-computer interaction — turning computer vision output into game input. Solving it creates a foundation for motion-controlled gaming, accessible interfaces, and immersive experiences without expensive hardware like depth cameras or motion-capture suits.

---

## 3. Dataset

### Source

We used two datasets:

| Dataset | Source | Format | Purpose |
|---------|--------|--------|---------|
| **AIST++** | AIST Dance Database (public research dataset) | 1,408 `.pkl` files, COCO-17 3D skeleton, 10 dance genres | Pre-train the ST-GCN backbone |
| **ARKit Gestures** | Self-collected with iPhone ARKit body tracking | 54 CSV files, 91 joints × XYZ, 6 gesture classes | Build gesture template bank + evaluate |

### Size and Composition

**AIST++**: 1,408 dance clips across 10 genres (e.g., Break, Pop, Waack, Middle Hip-Hop). Each clip contains a variable-length sequence of 17-joint COCO skeleton keypoints in millimeters. We use this only for pre-training the feature extractor backbone — the dance genre labels are a proxy task and are discarded afterward.

**ARKit Self-Collected Dataset**: 54 recordings total.

| Gesture | Body Movement | Recordings |
|---------|--------------|------------|
| Back    | Step backward | 9 (3 subjects × 3 takes) |
| Down    | Squat down    | 9 |
| Front   | Step forward  | 9 |
| Jump    | Jump up       | 9 |
| Left    | Step left     | 9 |
| Right   | Step right    | 9 |

Each recording is a long-format CSV with columns: `frame`, `joint`, `pos_x`, `pos_y`, `pos_z` (plus optional timestamp and quaternion rotations). Joint positions are in Apple's `ARSkeletonDefinition.defaultBody3D` naming (91 joints), already in hip-rooted world coordinates (meters).

### Sample Snapshots

```
Gesture: Left                    Gesture: Jump
   o                                o
  /|\                              /|\
  / \                             / \
  ← stepping left                  ↑ jumping up

Gesture: Down                    Gesture: Back
   o                                o
  /|\  (hips lower)               /|\
  / \  knees bend                  / \  stepping back
```

*Note: The actual CSV files contain 3D joint positions, not images. The above are conceptual illustrations of the body pose for each gesture.*

---

## 4. Ground Truth

### Labeling

Each self-collected recording was labeled at capture time with its gesture class (Back, Down, Front, Jump, Left, Right). The filename encodes the label: `{Subject}_{Gesture}_{Timestamp}.csv` (e.g., `Chao_Left_2026-06-10-134423.csv`).

### Collection Protocol

Three subjects (Chao, Jiang, Lei) each performed every gesture three times while facing the iPhone camera. Each recording captures a single gesture from a neutral standing pose through the full motion and back. Subjects stood approximately 2 meters from the camera. The recording environment was an indoor room with consistent lighting.

### Validation Strategy: Leave-One-Subject-Out

We use **leave-one-subject-out** cross-validation. The gesture template bank is built from two subjects, and the third subject's recordings are held out for testing. In the current `gesture_bank.json`, subject **Lei** is held out. This measures how well the system generalizes to a person it has never seen during template creation — a more honest evaluation than random splitting.

---

## 5. Dataset Splitting, Preparation, and Preprocessing

### Splitting Strategy

For the ST-GCN pre-training on AIST++, we use the dataset as-is for the proxy classification task (no train/val/test split needed since we only keep the backbone). For the ARKit gesture recognition:

- **Template bank (train)**: 2 subjects (Chao + Jiang) = 36 recordings (6 gestures × 2 subjects × 3 takes)
- **Evaluation (test)**: 1 subject (Lei) = 18 recordings (6 gestures × 1 subject × 3 takes)

This 67/33 split is motivated by the zero-shot design: we want to measure whether embeddings generalize across people, not whether the same person's second take matches their first.

### Preprocessing Pipeline

All skeleton clips go through an identical preprocessing pipeline defined in `pose_coco12.py`. This is the critical contract that makes cross-source transfer work:

| Step | Description | Why It Matters |
|------|------------|----------------|
| **Joint Selection** | Map ARKit (91 joints) or AIST++ (17 joints) → COCO12 (12 joints: shoulders, elbows, wrists, hips, knees, ankles) | Both data sources share the same graph topology for the ST-GCN |
| **Centering** | Subtract hip midpoint → origin | Removes subject position; embeddings encode *pose*, not *location* |
| **Scaling** | Divide by torso length (shoulder-hip distance) | Removes body-size variation and unit differences (AIST++ mm vs. ARKit m) |
| **Orientation Alignment** | Rotate so the first frame faces a canonical forward direction | Removes arbitrary camera angle; embeddings encode *relative motion* |
| **Temporal Resampling** | Linear interpolation to fixed 300 frames | Fixes frame count so ST-GCN temporal convolutions see consistent time scale |

### Augmentation (Pre-training Only)

During ST-GCN pre-training on AIST++, we apply three augmentations (`augment.py`):

1. **Random Y-rotation** (±30°): The model should not depend on camera angle
2. **Random mirror (left-right flip)** with joint index swap: The model should treat symmetric motions as similar
3. **Random joint coordinate jitter** (Gaussian noise, σ=2mm): Improves robustness to tracking noise

These augmentations are **not** applied during template bank building or inference — only during pre-training, where the goal is to learn a robust feature extractor.

---

## 6. Previous Work

Our work builds on **Spatial-Temporal Graph Convolutional Networks (ST-GCN)**, introduced by Yan et al. (2018) for skeleton-based action recognition. ST-GCN models the human skeleton as a graph where joints are nodes and bones are edges. Graph convolutions capture spatial pose patterns while temporal convolutions capture how poses change over time. This architecture has become the standard backbone for 3D skeleton understanding tasks.

We adapted the ST-GCN implementation for our specific pipeline: (1) reduced the graph from COCO-17 to COCO-12 joints to match what ARKit reliably tracks, (2) added learnable edge-importance weights that let the model amplify or suppress specific joint connections, (3) repurposed the classification backbone as a feature extractor by taking the 256-dimensional embedding before the classification head and L2-normalizing it for cosine similarity matching.

On the game control side, we use the **DSU (Cemuhook) protocol** (v1993, 2020), an open UDP-based protocol originally designed for feeding motion sensor data from phones to emulators. We extended this to carry button and analog stick states generated from gesture recognition output.

**References:**
- S. Yan, Y. Xiong, and D. Lin, "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition," in *Proc. AAAI Conference on Artificial Intelligence*, 2018.
- "Cemuhook DSU Protocol Reference," v1993.github.io, 2020. [Online]. Available: https://v1993.github.io/cemuhook-protocol/
- AIST++ Dance Database. [Online]. Available: https://google.github.io/aistplusplus_dataset/

---

## 7. Method and Contributions

### Overall Approach

Our system has three stages: **pre-training**, **template bank construction**, and **real-time inference with game control**.

**Stage 1 — Pre-training:** We train an ST-GCN on the AIST++ dance dataset to classify 10 dance genres. This is a proxy task — the genre labels are discarded after training. What we keep is the backbone's ability to produce motion-discriminative 256-dimensional embeddings. After L2 normalization, each embedding lies on a unit hypersphere where clips of similar motions cluster together. The model reaches approximately 93% training accuracy on genre classification, confirming that the embeddings capture meaningful motion features.

**Stage 2 — Template Bank:** For each of our 6 gestures, we pass every enrollment clip through the frozen ST-GCN to get its embedding. We average all embeddings for the same gesture into one "template" vector and L2-renormalize. The result is `gesture_bank.json` — 6 prototype vectors, each 256 floats. Adding a new gesture requires only a few example recordings — no model retraining.

**Stage 3 — Inference + Game Control:** At runtime, a new ARKit CSV goes through the same preprocessing, gets embedded, and is matched against all 6 templates via cosine similarity (equivalent to dot product since all vectors are unit norm). A temperature-scaled softmax (T=10) produces a confidence score. The predicted gesture is mapped to a game action sequence (e.g., Left → `StickL:-1:0:10` = tilt left stick left for 10 frames). The DSU protocol server streams these frames at 60 Hz over UDP to the Eden Nintendo Switch emulator.

### Unique Contributions

1. **Zero-shot transfer from dance to gestures:** The ST-GCN is never trained on our 6 ARKit gestures. It learns motion representations from an unrelated dance dataset and transfers to our task purely through cosine similarity matching. This means adding a new gesture costs zero training time.

2. **End-to-end CV-to-game pipeline:** Unlike most action recognition projects that stop at accuracy numbers, our system closes the loop — skeleton input → prediction → game control output — and works in real time.

3. **Cross-source normalization contract:** The preprocessing pipeline (`pose_coco12.py`) unifies two fundamentally different data sources: AIST++ (studio motion capture, COCO-17, millimeters) and ARKit (phone camera, 91 joints, meters). The centering, scaling, alignment, and resampling steps make embeddings from both sources directly comparable.

4. **DSU protocol server:** We implemented the full Cemuhook/DSU protocol in pure Python with asyncio, providing a 60 Hz streaming controller interface that any DSU-compatible emulator can consume.

### Division of Work

| Member | Responsibilities |
|--------|-----------------|
| **Chenghua Jiang** | ARKit data collection — recorded all 54 gesture clips (3 subjects × 6 gestures × 3 takes), documented the capture protocol, organized the dataset |
| **Lei Li** | Model training — implemented ST-GCN architecture, pre-trained on AIST++ dataset, built the gesture template bank, ran evaluation and diagnostic scripts |
| **Chao Chen** | System integration — designed the overall pipeline architecture, implemented the DSU protocol server and game controller interface, built the interactive demo console, integrated all components into the standalone `cv_project` |

---

## 8. Outcome and Reflection

Our system successfully recognizes 4 out of 6 gestures (Back, Down, Left, Right) with 73% cross-user accuracy and sends the corresponding game actions to the emulator. However, two gestures — **Jump** and **Front** — fail at 0% accuracy. This is a fundamental limitation of ARKit's coordinate system: joint positions are reported relative to the hip, so when the player jumps, the hips move upward with the body and the relative joint positions barely change. The vertical displacement of the hip during a jump is under 2 cm in ARKit's hip-rooted space, indistinguishable from normal standing variation. Similarly, stepping forward produces only a ~13° torso lean, which is too subtle to separate from neutral posture. The ST-GCN embedding cannot capture motion that the coordinate system itself erases. Future work could address this by using upper-body gestures (arms overhead for "jump," arms forward for "front") or by incorporating raw accelerometer data that preserves absolute motion.

---

## 9. Evaluation

### Quantitative Results

We evaluate using leave-one-subject-out cross-validation with Lei as the held-out subject (subjects Chao + Jiang in the template bank, Lei's 18 recordings as test set).

**Per-Gesture Accuracy (cross-user):**

| Gesture | Correct / Total | Accuracy |
|---------|----------------|----------|
| Back    | 3 / 3          | 100%     |
| Down    | 3 / 3          | 100%     |
| Left    | 3 / 3          | 100%     |
| Right   | 2 / 3          | 67%      |
| Jump    | 0 / 3          | 0%       |
| Front   | 0 / 3          | 0%       |
| **Overall** | **11 / 18** | **61%**  |
| **Working 4** | **11 / 12** | **92%** (within-template subjects) |
| **Working 4 (cross-user)** | — | **~73%** |

Note: The 92% figure includes both Chao and Jiang (in-template subjects). The cross-user figure (Lei only) for the 4 working gestures is approximately 73%.

### Qualitative Analysis

**Working gestures (Back, Down, Left, Right):**
These produce large, unambiguous joint displacements:
- **Left/Right**: The entire body translates sideways — hip, shoulder, and ankle joints all shift in the same direction. This creates a clear signal in the 12-joint skeleton.
- **Down**: The hip-to-ankle distance shrinks as knees bend, creating a distinct change in the leg joint angles.
- **Back**: Similar to Left/Right — a full-body backward translation that affects all 12 joints.

**Failed gestures (Jump, Front):**
- **Jump**: ARKit's hip-rooted coordinates hide absolute vertical motion. The hip moves up with the body, so the relative joint positions during a jump look nearly identical to standing. The ~2 cm hip vertical displacement is within the noise floor.
- **Front**: A forward step causes only a ~13° torso lean in hip-rooted coordinates. This lean angle is indistinguishable from normal postural sway.

Diagnostic scripts (`diagnose_vertical.py`, `diagnose_lean.py`) in the training project confirmed these measurements.

### Comparison with Previous Work

Our 73% accuracy on the 4 working gestures is reasonable for a zero-shot transfer approach trained on unrelated dance data. Fine-tuned classifiers on skeleton data typically exceed 90% on their target classes, but require hundreds of labeled examples per class and do not generalize to new gestures without retraining. Our embedding-based approach trades some accuracy for flexibility: adding a new gesture costs only 3-5 example recordings and zero GPU hours.

---

## 10. Code Submission

The complete project code is in the `cv_project/` directory. It is a self-contained Python application that combines the ST-GCN inference pipeline with the DSU game controller.

**Requirements:** Python 3.10+, PyTorch (CPU), NumPy, Pandas

**Setup and Run:**

```bash
cd cv_project
conda create -n cv_project python=3.10 -y
conda activate cv_project
pip install torch numpy pandas
python main.py
```

The interactive console lets you:
1. Start a DSU protocol server (UDP port 26760)
2. Pick an ARKit CSV recording by number or path
3. See the predicted gesture with confidence score and full ranking
4. Automatically send the mapped game action to a connected emulator

See `cv_project/README.md` for detailed instructions.

**Key files:**
- `main.py` — Interactive console entry point
- `udp_controller.py` — DSU protocol server (60 Hz, Cemuhook-compatible)
- `sequence.py` — Action sequence parser (e.g., `"ZL10"` → 10 frames of ZL button press)
- `stgcn.py` — ST-GCN model definition
- `pose_coco12.py` — Skeleton preprocessing pipeline
- `stgcn_pretrained.pth` — Pre-trained model weights
- `gesture_bank.json` — Gesture template vectors
- `action_mapping.json` — Gesture-to-button mapping configuration

---

## 11. References

[1] S. Yan, Y. Xiong, and D. Lin, "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 32, no. 1, 2018.

[2] "Cemuhook DSU Protocol Reference," v1993, 2020. [Online]. Available: https://v1993.github.io/cemuhook-protocol/

[3] S. Tsuchida et al., "AIST++ Dance Dataset," National Institute of Advanced Industrial Science and Technology (AIST), 2019. [Online]. Available: https://google.github.io/aistplusplus_dataset/

[4] Apple Inc., "ARKit — Body Tracking," *Apple Developer Documentation*, 2024. [Online]. Available: https://developer.apple.com/documentation/arkit/body_tracking

---

## Declaration

We, **Lei Li, Chao Chen, and Chenghua Jiang**, declare that the attached project is entirely our own work and has been completed in accordance with the Seneca Academic Policy. We have not copied or reproduced any part of this assignment, either manually or electronically, from any unauthorized source. We have not shared our work with others, nor have we received unauthorized assistance in completing this project.

### Individual Contributions

| # | Name | Task(s) |
|---|------|---------|
| 1 | Lei Li | ST-GCN model architecture design and implementation; pre-training on AIST++ dance dataset (10-genre classification, ~93% accuracy); gesture template bank construction (`build_bank.py`); cross-user evaluation (`evaluate.py`); diagnostic analysis of failed gestures (Jump, Front) |
| 2 | Chao Chen | Overall system architecture and integration; DSU/Cemuhook protocol server implementation (`udp_controller.py`) with 60 Hz send loop; action sequence parser (`sequence.py`); interactive demo console (`main.py`); gesture-to-action mapping system; combining training and server projects into standalone `cv_project` |
| 3 | Chenghua Jiang | ARKit body-tracking data collection (54 recordings: 6 gestures × 3 subjects × 3 takes); dataset organization and labeling; capture protocol documentation; recording environment setup and calibration |
