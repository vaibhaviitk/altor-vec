#!/usr/bin/env python3
"""Encode docs.json with sentence-transformers → data/embeddings.bin

Binary format: u32 LE count + u32 LE dims + count*dims f32 LE values
"""

import json
import os
import struct

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DOCS_PATH = os.path.join(DATA_DIR, "docs.json")
OUT_PATH = os.path.join(DATA_DIR, "embeddings.bin")
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256


def main():
    with open(DOCS_PATH) as f:
        docs = json.load(f)

    texts = [d["text"] for d in docs]
    print(f"Encoding {len(texts)} passages with {MODEL_NAME}...")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32)

    count, dims = embeddings.shape
    print(f"Embeddings shape: {count} x {dims}")

    with open(OUT_PATH, "wb") as f:
        f.write(struct.pack("<II", count, dims))
        f.write(embeddings.tobytes())

    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"Wrote {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
