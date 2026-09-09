"""Convert HF parquet shards of 140k Real and Fake Faces to an ImageFolder tree.

Usage:
    python scripts/prepare_data.py

Reads data/raw_parquet/{train,validation,test}-*.parquet and writes
dataset/{train,val,test}/{real,fake}/*.jpg
Label convention (from the HF dataset card): 0 = real, 1 = fake.
"""
import glob
import io
import os
import sys

import pyarrow.parquet as pq
from PIL import Image

RAW_DIR = "data/raw_parquet"
OUT_DIR = "dataset"
SPLIT_MAP = {"train": "train", "validation": "val", "test": "test"}
LABEL_MAP = {0: "real", 1: "fake"}

# Cap per split/class so a laptop can train end-to-end quickly.
MAX_PER_CLASS = {"train": None, "val": None, "test": None}


def iter_rows(parquet_path):
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=256):
        cols = batch.to_pydict()
        for img, label in zip(cols["image"], cols["label"]):
            yield img, label


def main():
    total = 0
    for shard_prefix, split_name in SPLIT_MAP.items():
        counts = {"real": 0, "fake": 0}
        shards = sorted(glob.glob(os.path.join(RAW_DIR, f"{shard_prefix}-*.parquet")))
        if not shards:
            print(f"[skip] no shards for '{shard_prefix}'", file=sys.stderr)
            continue
        out_split = os.path.join(OUT_DIR, split_name)
        os.makedirs(os.path.join(out_split, "real"), exist_ok=True)
        os.makedirs(os.path.join(out_split, "fake"), exist_ok=True)

        for shard in shards:
            for i, (img_rec, label) in enumerate(iter_rows(shard)):
                cls = LABEL_MAP[label]
                cap = MAX_PER_CLASS.get(split_name)
                if cap is not None and counts[cls] >= cap:
                    continue
                raw = img_rec["bytes"]
                image = Image.open(io.BytesIO(raw)).convert("RGB")
                fname = f"{split_name}_{cls}_{counts[cls]:06d}.jpg"
                image.save(os.path.join(out_split, cls, fname), quality=92)
                counts[cls] += 1
                total += 1
                if total % 2000 == 0:
                    print(f"  wrote {total} images...", flush=True)
        print(f"{split_name}: {counts}", flush=True)
    print(f"done — {total} images written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
