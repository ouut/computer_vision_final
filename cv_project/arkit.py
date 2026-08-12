"""Read ARKit body-tracking CSV recordings.

Long-format CSV with columns: frame, joint, pos_x, pos_y, pos_z (plus an
optional timestamp and rotation quaternions, which are ignored). Positions are
already in a common hip-rooted space, so they are used directly.
"""
import numpy as np
import pandas as pd

from pose_coco12 import ARKIT_TO_COCO12


def load_arkit_csv(path):
    """Return {joint_name: (T, 3)} for the joints needed by the COCO-12 map."""
    df = pd.read_csv(path, sep=None, engine="python")  # sniff the delimiter
    df.columns = [str(c).strip() for c in df.columns]

    if "frame" in df.columns:
        frames = [g for _, g in df.groupby("frame", sort=True)]
    else:
        first = df["joint"].iloc[0]
        starts = df.index[df["joint"] == first].tolist() + [len(df)]
        frames = [df.iloc[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]

    needed = list(ARKIT_TO_COCO12.values())
    T = len(frames)
    out = {j: np.zeros((T, 3), dtype=np.float32) for j in needed}
    for t, frame in enumerate(frames):
        pos = frame.set_index("joint")[["pos_x", "pos_y", "pos_z"]]
        for j in needed:
            if j not in pos.index:
                raise KeyError(f"frame {t} is missing joint {j}")
            out[j][t] = pos.loc[j].to_numpy(dtype=np.float32)
    return out
