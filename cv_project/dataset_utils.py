"""Helpers for the folder-organised ARKit gesture dataset.

Expected layout (gesture = folder name, subject parsed from filename prefix):

    dataset/
        Jump/
            Lei_Jump_2026-06-10-135221.csv
            Chao_Jump_...csv
        Down/
            ...
"""
import glob
import os

import numpy as np
import torch

from arkit import load_arkit_csv
from pose_coco12 import arkit_to_coco12, preprocess
from stgcn import clips_to_tensor


def subject_of(path):
    """Subject name = text before the first underscore (e.g. 'Lei')."""
    return os.path.basename(path).split("_", 1)[0]


def list_samples(root):
    """Return [(path, gesture, subject), ...] for every csv under root."""
    samples = []
    for gesture in sorted(os.listdir(root)):
        gdir = os.path.join(root, gesture)
        if not os.path.isdir(gdir):
            continue
        for path in sorted(glob.glob(os.path.join(gdir, "*.csv"))):
            samples.append((path, gesture, subject_of(path)))
    return samples


def embed_file(model, path, target_len=300, device="cpu", do_orientation=True):
    """Load one csv and return its (256,) embedding as a numpy array."""
    joints = load_arkit_csv(path)
    clip = preprocess(arkit_to_coco12(joints),
                      do_orientation=do_orientation, target_len=target_len)
    x = clips_to_tensor(clip).to(device)
    return model.get_embedding(x)[0].cpu().numpy()
