"""
Builds the FAISS vector store from the customer support dataset.

The free-tier Gemini embedding quota is 1000 requests per DAY and every row costs one
request, so the full 26872-row dataset cannot be indexed in one go (it would take ~27
days). Instead this script samples an even number of rows per intent, which keeps all 27
intents reachable by the retriever while staying inside one day's quota.

Usage:
    python build_index.py                  # 15 rows per intent  -> 405 vectors
    python build_index.py --per-intent 35  # 35 rows per intent  -> 945 vectors
    python build_index.py --all            # everything (needs many days of quota)
"""

import argparse
import os

import pandas as pd

from src.helper import EMBED_REQUESTS_PER_DAY, create_embeddings, create_index

DATASET_PATH = "Customer_Support_Training_Dataset/Customer_Support_Training_Dataset.csv"
INDEX_PATH = "vector_store/faiss_index.index"
#the rows that were actually indexed, so demo.py looks up the same rows the index holds
INDEXED_ROWS_PATH = "vector_store/indexed_rows.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-intent", type=int, default=15, help="rows to sample per intent")
    parser.add_argument("--all", action="store_true", help="index the whole dataset")
    parser.add_argument("--seed", type=int, default=42, help="sampling seed")
    args = parser.parse_args()

    df = pd.read_csv(DATASET_PATH)
    #near-duplicate instructions waste quota without adding retrieval coverage
    df = df.drop_duplicates(subset="instruction").reset_index(drop=True)
    print(f"Dataset: {len(df)} unique instructions across {df['intent'].nunique()} intents.")

    if args.all:
        sample = df
    else:
        #built explicitly rather than with groupby().apply(): pandas 3.0 drops the
        #grouping column from the frames it passes to apply, which silently loses `intent`
        groups = [
            group.sample(min(len(group), args.per_intent), random_state=args.seed)
            for _, group in df.groupby("intent", sort=True)
        ]
        sample = pd.concat(groups).reset_index(drop=True)

    print(f"Selected {len(sample)} rows to embed.")
    if len(sample) > EMBED_REQUESTS_PER_DAY:
        days = -(-len(sample) // EMBED_REQUESTS_PER_DAY)
        print(
            f"WARNING: {len(sample)} rows exceeds the free-tier daily cap of "
            f"{EMBED_REQUESTS_PER_DAY}. This needs ~{days} days of quota and will stall "
            f"with 429s. Lower --per-intent or upgrade to a paid tier."
        )
        if input("Continue anyway? [y/N] ").strip().lower() != "y":
            return

    print("Embedding (this takes a few minutes - the free tier is rate limited)...")
    vectors = create_embeddings(sample, column_name="instruction")
    print(f"Embeddings shape: {vectors.shape}")

    create_index(vectors, INDEX_PATH)

    os.makedirs(os.path.dirname(INDEXED_ROWS_PATH), exist_ok=True)
    sample.to_csv(INDEXED_ROWS_PATH, index=False)
    print(f"Indexed rows saved to {INDEXED_ROWS_PATH}. You can now run: streamlit run demo.py")


if __name__ == "__main__":
    main()
