#!/usr/bin/env python3
"""Download 10K unique MS MARCO passages → data/docs.json"""

import json
import os
from datasets import load_dataset

NUM_PASSAGES = 10_000
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH = os.path.join(OUT_DIR, "docs.json")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Streaming MS MARCO v2.1 passages, collecting {NUM_PASSAGES} unique...")
    ds = load_dataset(
        "microsoft/ms_marco", "v2.1", split="train", streaming=True
    )

    seen = set()
    docs = []

    for row in ds:
        for passage in row["passages"]["passage_text"]:
            text = passage.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            docs.append({"id": len(docs), "text": text})
            if len(docs) % 1000 == 0:
                print(f"  collected {len(docs)}/{NUM_PASSAGES}")
            if len(docs) >= NUM_PASSAGES:
                break
        if len(docs) >= NUM_PASSAGES:
            break

    with open(OUT_PATH, "w") as f:
        json.dump(docs, f)

    print(f"Wrote {len(docs)} passages to {OUT_PATH}")


if __name__ == "__main__":
    main()
