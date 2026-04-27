import os
import re
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


DATA_PATH = Path("data/lex_fridman_podcast.parquet")
CHROMA_DIR = Path("chroma_db")
COLLECTION = "lex_fridman"

# 80 sentences ~= 700-1000 words per chunk, 10 overlap keeps continuity
SENTENCES_PER_CHUNK = 80
OVERLAP = 10

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def episode_number(title: str) -> int:
    # titles look like "Max Tegmark: Life 3.0 | Lex Fridman Podcast #1"
    m = re.search(r"#(\d+)", title)
    return int(m.group(1)) if m else -1


def make_chunks(df_ep):
    rows = df_ep.to_dict("records")
    n = len(rows)
    step = SENTENCES_PER_CHUNK - OVERLAP
    out = []
    for start in range(0, n, step):
        block = rows[start : start + SENTENCES_PER_CHUNK]
        if len(block) < 5:
            break
        text = " ".join(r["text"].strip() for r in block if r["text"])
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 50:
            continue
        out.append({
            "text": text,
            "source_url": block[0]["timestamp_link"],
            "start_idx": start,
        })
    return out


def main(limit_episodes=None):
    print(f"loading {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print("rows:", len(df), "episodes:", df["episode"].nunique())

    eps = sorted(df["episode"].unique(), key=episode_number)
    if limit_episodes:
        eps = eps[:limit_episodes]
        print(f"using first {limit_episodes} episodes for demo")

    docs = []
    for ep in eps:
        ep_df = df[df["episode"] == ep]
        ep_num = episode_number(ep)
        # guest = bit before the first colon, usually
        guest = ep.split(":")[0].strip()
        for c in make_chunks(ep_df):
            docs.append(Document(
                page_content=c["text"],
                metadata={
                    "episode": ep,
                    "episode_number": ep_num,
                    "guest": guest,
                    "source_url": c["source_url"],
                    "chunk_idx": c["start_idx"],
                },
            ))

    print(f"built {len(docs)} chunks")

    embed = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # blow away any old index so we don't double-insert
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)

    print("embedding & writing to chroma...")
    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=embed,
        persist_directory=str(CHROMA_DIR),
    )

    # add in batches so we get progress and don't blow memory
    BATCH = 256
    for i in range(0, len(docs), BATCH):
        vs.add_documents(docs[i : i + BATCH])
        print(f"  {min(i+BATCH, len(docs))}/{len(docs)}")

    print("done. chroma persisted at", CHROMA_DIR)


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit_episodes=n)
