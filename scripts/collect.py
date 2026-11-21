"""Collect expert (DART-noised) rollouts.  Usage: python scripts/collect.py N_EPISODES OUT.npz [NOISE]"""

import sys
from multiprocessing import Pool

sys.path.insert(0, ".")
import numpy as np

from navfm.data import merge, rollout

TRAIN_SEED0 = 100_000  # training seeds never overlap the evaluation seeds (0..999)


def job(i):
    noise = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    return rollout(TRAIN_SEED0 + i, noise=noise if i % 4 else 0.0)


if __name__ == "__main__":
    n = int(sys.argv[1])
    with Pool(4) as p:
        eps = p.map(job, range(n), chunksize=8)
    d = merge(eps)
    print("samples", len(d["act"]), "expert-executed success", np.mean([e["success"] for e in eps]))
    np.savez_compressed(sys.argv[2], **d)
