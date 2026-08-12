#!/usr/bin/env python3
"""
MotionPlay — Gesture-to-Game Interactive Console

Start a DSU server, then interactively pick ARKit CSV files to classify.
Each prediction is mapped to a game action and sent to the emulator via UDP.
Loop until you press 'q'.

Usage:
    python main.py
    python main.py --frame-count 20 --game-port 26760
"""

import argparse
import asyncio
import glob
import json
import os
import time

import numpy as np
import torch

from pose_coco12 import build_adjacency
from stgcn import STGCN
from dataset_utils import embed_file
from udp_controller import DSUServer

# ═══════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════
DEFAULT_WEIGHTS = "stgcn_pretrained.pth"
DEFAULT_BANK = "gesture_bank.json"
DEFAULT_MAPPING = "action_mapping.json"
DEFAULT_FRAME_COUNT = 20
DEFAULT_LENGTH = 300
DSU_HOST = "0.0.0.0"
DSU_PORT = 26760
DATASET_DIR = "dataset"


# ═══════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════
def load_predictor(weights, bank_path, device="cpu"):
    """Load ST-GCN model and gesture template bank.

    Returns (model, gesture_names, template_matrix, do_orientation).
    """
    with open(bank_path) as f:
        bank = json.load(f)
    gestures = bank["gestures"]
    templates = np.array([bank["templates"][g] for g in gestures], dtype=np.float32)
    do_orientation = bank.get("do_orientation", True)

    ckpt = torch.load(weights, map_location=device)
    num_class = len(ckpt.get("genres", [0] * 10))
    model = STGCN(build_adjacency(), num_class=num_class).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, gestures, templates, do_orientation


def predict_one(model, gestures, templates, csv_path,
                do_orientation=True, length=300, device="cpu"):
    """Classify a single CSV clip.

    Returns (predicted_gesture, confidence, ranking_list).
    """
    emb = embed_file(model, csv_path, length, device, do_orientation=do_orientation)
    sims = templates @ emb                          # cosine similarity

    exp = np.exp((sims - sims.max()) * 10.0)        # temperature-scaled softmax
    probs = exp / exp.sum()

    order = np.argsort(-sims)
    ranking = [(gestures[i], float(sims[i]), float(probs[i])) for i in order]
    best = ranking[0]
    return best[0], best[2], ranking


# ═══════════════════════════════════════════════════════════
# Action mapping
# ═══════════════════════════════════════════════════════════
def load_action_mapping(path):
    """Load gesture → action-sequence mapping from JSON file."""
    with open(path) as f:
        return json.load(f)


def build_sequence(gesture, mapping, frame_count):
    """Resolve a gesture to its action sequence string, or None if unmapped."""
    if gesture not in mapping:
        return None
    return mapping[gesture].replace("{n}", str(frame_count))


# ═══════════════════════════════════════════════════════════
# CSV scanner
# ═══════════════════════════════════════════════════════════
def scan_csv_files(data_dir):
    """Return list of (gesture, path) sorted by gesture then filename."""
    files = []
    if not os.path.isdir(data_dir):
        return files
    for gesture in sorted(os.listdir(data_dir)):
        gdir = os.path.join(data_dir, gesture)
        if not os.path.isdir(gdir):
            continue
        for path in sorted(glob.glob(os.path.join(gdir, "*.csv"))):
            files.append((gesture, path))
    return files


# ═══════════════════════════════════════════════════════════
# Interactive console
# ═══════════════════════════════════════════════════════════
def _client_status(dsu):
    clients = getattr(dsu, "_clients", {})
    if clients:
        return f"{len(clients)} connected"
    return "waiting"


async def _monitor_connection(dsu):
    """Background task: print a message when the emulator connects or disconnects.
    Runs while input() is blocking, so the user gets immediate feedback.
    """
    was_empty = True
    while True:
        await asyncio.sleep(0.5)
        is_empty = not dsu._clients
        if not is_empty and was_empty:
            addr = list(dsu._clients.keys())[0]
            print(f"\n  ✓ Emulator connected from {addr[0]}:{addr[1]}\n")
        elif is_empty and not was_empty:
            print("\n  ✗ Emulator disconnected\n")
        was_empty = is_empty


def print_banner(gestures, dsu_host, dsu_port, frame_count, data_dir):
    print()
    print("=" * 62)
    print("  MotionPlay — Gesture-to-Game  Interactive Console")
    print("=" * 62)
    print(f"  gestures:     {len(gestures)} ({', '.join(gestures)})")
    print(f"  DSU server:   udp://{dsu_host}:{dsu_port}")
    print(f"  frame-count:  {frame_count}  "
          f"({frame_count / 60:.2f}s @ 60Hz)")
    print(f"  dataset:      {data_dir}/")
    print("-" * 62)
    print("  number       pick a CSV from the list")
    print("  list         show the file list again")
    print("  /path/to.csv classify a custom file")
    print("  q            quit")
    print()


def print_file_list(files):
    """Print numbered file list grouped by gesture."""
    if not files:
        print("  (no CSV files found in dataset/)\n")
        return
    print(f"\n  {'#':>3s}  {'Label':8s}  File")
    print(f"  {'-' * 3}  {'-' * 8}  {'-' * 45}")
    for i, (gesture, path) in enumerate(files):
        fname = os.path.basename(path)
        print(f"  {i:3d}  {gesture:8s}  {fname}")
    print()


def print_prediction(gesture, confidence, ranking):
    """Print prediction result — matches predict.py output style."""
    print(f"\n  prediction: {gesture}   (confidence {confidence:.1%})\n")
    print("  ranking:")
    for name, sim, prob in ranking:
        bar = "#" * int(prob * 30)
        print(f"    {name:8s}  cos={sim:+.3f}  {prob:5.1%}  {bar}")
    print()


def print_action(gesture, seq, frame_count, dsu):
    """Print the action being sent to the emulator."""
    if seq is None:
        print(f"  [warn]  no action mapped for '{gesture}'")
        print(f"  [warn]  edit action_mapping.json to add one\n")
        return
    print(f"  ▶ action:    {seq}")
    print(f"    duration:   {frame_count} frames "
          f"({frame_count / 60:.2f}s @ 60Hz)")
    clients = getattr(dsu, "_clients", {})
    if clients:
        for addr in clients:
            print(f"    sending to  {addr[0]}:{addr[1]}")
    else:
        print(f"    [warn]  no emulator — frames dropped")
    print()


async def run_interactive(model, gestures, templates, do_orient, mapping,
                          dsu, frame_count, length, device, data_dir):
    """Main interactive loop — runs until user types 'q'."""
    files = scan_csv_files(data_dir)
    first_run = True

    # Background task: notify when emulator connects / disconnects
    monitor_task = asyncio.create_task(_monitor_connection(dsu))

    try:
      while True:
        if first_run:
            print_banner(gestures, dsu.bind_host, dsu.port,
                         frame_count, data_dir)
            first_run = False

        # Non-blocking input — event loop stays alive for DSU sends
        try:
            choice = (await asyncio.to_thread(
                input, f"csv [{_client_status(dsu)}] > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if choice == "":
            continue

        if choice.lower() == "q":
            print("quit")
            break

        if choice.lower() in ("l", "list"):
            print_file_list(files)
            continue

        if choice.lower() in ("h", "help"):
            print("  number       pick CSV from the list above")
            print("  list         show the file list")
            print("  /path/to.csv classify a custom CSV file")
            print("  q            quit\n")
            continue

        # ── Resolve the CSV path ──────────────────────────
        csv_path = None
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(files):
                csv_path = files[idx][1]
            else:
                print(f"  [warn]  {idx} is out of range "
                      f"(0–{len(files) - 1})\n")
                continue
        else:
            # Treat as a file path
            if os.path.exists(choice):
                csv_path = choice
            else:
                print(f"  [warn]  file not found: {choice}\n")
                continue

        # ── Predict ───────────────────────────────────────
        t0 = time.time()
        try:
            gesture, confidence, ranking = predict_one(
                model, gestures, templates, csv_path,
                do_orientation=do_orient, length=length, device=device)
        except Exception as exc:
            print(f"  [error] prediction failed: {exc}\n")
            continue

        elapsed = time.time() - t0

        print(f"\n  file: {csv_path}")
        print_prediction(gesture, confidence, ranking)

        # ── Map & send action ─────────────────────────────
        seq = build_sequence(gesture, mapping, frame_count)
        print_action(gesture, seq, frame_count, dsu)

        if seq:
            dsu.push_sequence(seq)
            # Let event loop deliver frames
            await asyncio.sleep(0.05)
            while dsu._buffer_pos < len(dsu._buffer):
                await asyncio.sleep(0.02)

        # Show file list after each prediction for convenience
        print_file_list(files)
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


# ═══════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════
async def main():
    parser = argparse.ArgumentParser(
        description="MotionPlay — Gesture-to-Game Interactive Console")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--bank", default=DEFAULT_BANK)
    parser.add_argument("--mapping", default=DEFAULT_MAPPING)
    parser.add_argument("--game-host", default=DSU_HOST)
    parser.add_argument("--game-port", type=int, default=DSU_PORT)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--dataset", default=DATASET_DIR)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────────────────────
    print(f"Loading model ({args.weights}) ...")
    model, gestures, templates, do_orient = load_predictor(
        args.weights, args.bank, device)
    print(f"  device: {device}  |  "
          f"{len(gestures)} gestures: {', '.join(gestures)}")

    # ── Load mapping ──────────────────────────────────────
    mapping = load_action_mapping(args.mapping)
    mapped = [g for g in gestures if g in mapping]
    print(f"  {len(mapped)}/{len(gestures)} gestures mapped to actions")

    # ── Start DSU server ──────────────────────────────────
    dsu = DSUServer(bind_host=args.game_host, port=args.game_port)
    await dsu.start()

    # ── Run interactive loop ──────────────────────────────
    try:
        await run_interactive(
            model, gestures, templates, do_orient, mapping,
            dsu, args.frame_count, args.length, device, args.dataset)
    finally:
        await dsu.stop()
        print("Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())
