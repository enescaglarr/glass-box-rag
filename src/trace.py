"""
Runs one query through the RAG pipeline and records every intermediate step.

Both entry points use this: the Streamlit app shows the trace live under the answer,
and make_report.py records a batch of them into a static page. Keeping it in one place
means the two can never drift apart.
"""

import re
import time
from typing import Any, Dict, List

import faiss
import numpy as np
import pandas as pd

from src.helper import LLM_MODEL, build_prompt, call_llm, embed_texts, prompt_parts

def parse_output(raw: str) -> Dict[str, Any]:
    """Splits the model's three numbered answers into separate fields."""
    urgency = re.search(r"^\s*1\.\s*(\d)", raw, re.M)
    category = re.search(r"^\s*2\.\s*(.+)$", raw, re.M)
    reply = re.search(r"^\s*3\.\s*(.*)$", raw, re.M | re.S)
    return {
        "urgency": int(urgency.group(1)) if urgency else None,
        "category_out": category.group(1).strip().rstrip(".") if category else None,
        #the model sometimes emits literal \n escapes inside its own text
        "reply": reply.group(1).strip().replace("\\n", "\n") if reply else raw,
    }


#Retrieval confidence bands, measured over the queries in make_report.py plus a set of
#deliberately out-of-domain ones. In-domain best matches land at 0.83-0.89; queries the
#dataset has no answer for land at 0.71-0.77.
CONFIDENCE_HIGH = 0.84
CONFIDENCE_LOW = 0.78

#Two same-intent rows can sit within noise of each other in cosine while their responses
#differ hugely in usefulness - one a 146-character stub, the other a 1153-character
#step-by-step guide. Below this gap, treat the scores as tied and prefer the fuller one.
COSINE_TIE = 0.01


def confidence(best_cos: float) -> str:
    """Buckets the best match into high / medium / low, for the reviewer."""
    if best_cos >= CONFIDENCE_HIGH:
        return "high"
    return "medium" if best_cos >= CONFIDENCE_LOW else "low"


def select_hits(candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Narrows an over-fetched candidate list down to the k rows actually worth sending.

    Two passes. First, at most one row per intent, so a compound query ("cancel my order
    and update my email") cannot have all its slots eaten by near-duplicates of a single
    intent. Second, among rows whose scores are tied within COSINE_TIE, prefer the longer
    response - cosine ranks how similar the *question* is and says nothing about whether
    the *answer* is any use.
    """
    #tie-break first, so the row that survives deduplication is the useful one
    ordered = sorted(candidates, key=lambda c: (-round(c["cos"] / COSINE_TIE), -len(c["response"])))

    seen, picked = set(), []
    for candidate in ordered:
        if candidate["intent"] not in seen:
            seen.add(candidate["intent"])
            picked.append(candidate)
        if len(picked) == k:
            break

    #a query genuinely about one intent may not have k distinct intents available
    if len(picked) < k:
        picked += [c for c in ordered if c not in picked][:k - len(picked)]

    return sorted(picked, key=lambda c: -c["cos"])


def trace_query(
    query: str,
    index: faiss.Index,
    df: pd.DataFrame,
    k: int = 3,
    overfetch: int = 6,
) -> Dict[str, Any]:
    """
    Answers one query and records how it got there.

    Args:
        query: the customer's inbound query.
        index: the FAISS index built by build_index.py.
        df: the rows that index was built from, in the same order.
        k: how many rows to hand to the model.
        overfetch: how many neighbours to pull before deduplicating by intent.

    Returns:
        A dict with the query vector's head and norm, the retrieved rows with their
        distances and cosine similarities, the assembled prompt, the raw model output,
        its three parsed fields, and per-stage timings.
    """
    t0 = time.time()
    query_vector = embed_texts([query], task_type="retrieval_query", show_progress=False)
    t_embed = time.time() - t0

    t1 = time.time()
    #over-fetch, then narrow - see select_hits
    distances, indices = index.search(query_vector, min(overfetch, index.ntotal))
    t_search = time.time() - t1

    candidates = []
    for distance, position in zip(distances[0], indices[0]):
        row = df.iloc[int(position)]
        distance = float(distance)
        candidates.append({
            "distance": round(distance, 4),
            #vectors are unit length, so cos = 1 - d^2/2
            "cos": round(1 - (distance ** 2) / 2, 4),
            "instruction": row["instruction"],
            "intent": row["intent"],
            "category": row["category"],
            "flags": row["flags"],
            "response": row["response"],
        })

    retrieved = select_hits(candidates, k)
    dropped = [c for c in candidates if c not in retrieved]
    responses = [hit["response"] for hit in retrieved]

    t2 = time.time()
    output = call_llm(query, responses)
    t_llm = time.time() - t2

    prompt = build_prompt(query, responses)

    trace = {
        "query": query,
        "query_vector_head": [round(float(x), 4) for x in query_vector[0][:8]],
        "vector_norm": round(float(np.linalg.norm(query_vector[0])), 4),
        "retrieved": retrieved,
        "prompt": prompt,
        "prompt_parts": prompt_parts(query, responses),
        "prompt_chars": len(prompt),
        "output": output.strip(),
        "llm_model": LLM_MODEL.replace("models/", ""),
        "dropped": dropped,
        "intents_hit": sorted({r["intent"] for r in retrieved}),
        "best_cos": max(r["cos"] for r in retrieved),
        "confidence": confidence(max(r["cos"] for r in retrieved)),
        "timing": {
            "embed": round(t_embed, 3),
            "search": round(t_search, 4),
            "llm": round(t_llm, 3),
            "total": round(t_embed + t_search + t_llm, 3),
        },
    }
    trace.update(parse_output(output))
    return trace
