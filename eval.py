"""
Repeatable evaluation of the retrieval and generation pipeline.

Runs a fixed set of queries whose correct intents are known, and scores four things the
manual testing in demo.md found worth watching:

  Retrieval   did the right intent come back at all, and was it ranked first?
  Confidence  does best-match cosine separate answerable queries from unanswerable ones?
  Urgency     does the score track how fast a human must act, or just how much work?
  Hygiene     any {{placeholders}} left in the reply, any reply in the wrong language?

Usage:
    python eval.py                 # run everything, print a report
    python eval.py --json out.json # also write the raw results
    python eval.py --only T1 T6    # run named cases while iterating
"""

import argparse
import json
import re
import time

import faiss
import pandas as pd

from src.trace import CONFIDENCE_HIGH, CONFIDENCE_LOW, trace_query

INDEX_PATH = "vector_store/faiss_index.index"
INDEXED_ROWS_PATH = "vector_store/indexed_rows.csv"

#`expect` is the set of intents any of which counts as a correct match; an empty set
#means the dataset has no answer for this query and the pipeline should say so.
#`urgency` is the band a sensible human reviewer would accept, inclusive.
CASES = [
    {
        "id": "T1", "query": "I need to change the shipping address on my order",
        "expect": {"change_shipping_address", "set_up_shipping_address"},
        "urgency": (2, 3), "language": "en",
        "note": "Clean in-domain baseline",
    },
    {
        "id": "T2", "query": "cancel order",
        "expect": {"cancel_order"},
        "urgency": (2, 3), "language": "en",
        "note": "Keyword mode, no sentence structure",
    },
    {
        "id": "T3", "query": "helo i cnt log in to my acount pls help",
        "expect": {"recover_password", "registration_problems"},
        "urgency": (3, 5), "language": "en",
        "note": "Heavy typos",
    },
    {
        "id": "T4", "query": "WHY HAVE I STILL NOT RECEIVED MY REFUND?!?! this is ridiculous",
        "expect": {"track_refund", "get_refund"},
        "urgency": (4, 5), "language": "en",
        "note": "Anger and caps, genuinely urgent",
    },
    {
        "id": "T5", "query": "do you have this in size medium",
        "expect": set(),
        "urgency": (1, 3), "language": "en",
        "note": "Out of domain - no stock intent exists",
    },
    {
        "id": "T6", "query": "siparişimi iptal etmek istiyorum",
        "expect": {"cancel_order"},
        "urgency": (2, 3), "language": "tr",
        "note": "Turkish, cross-lingual retrieval",
    },
    {
        "id": "T7", "query": "it doesn't work",
        "expect": set(),
        "urgency": (1, 4), "language": "en",
        "note": "Too vague to answer",
    },
    {
        "id": "T8",
        "query": "i want to cancel my order and also update my email and check my invoice",
        "expect": {"cancel_order"},
        "urgency": (2, 3), "language": "en",
        "note": "Three intents in one message",
    },
    {
        "id": "T9", "query": "what payment methods do you accept",
        "expect": {"check_payment_methods"},
        "urgency": (1, 2), "language": "en",
        "note": "Purely informational, should score low urgency",
    },
    {
        "id": "T10", "query": "i was charged twice for the same order",
        "expect": set(),
        "urgency": (4, 5), "language": "en",
        #Not a pipeline failure: the corpus has no double-billing intent. Every
        #payment_issue row is about a payment that failed, not one that went through
        #twice, and across the full 26,872 rows "charge" almost always means a
        #cancellation fee. The right behaviour here is low confidence, not a match.
        "note": "Money at stake, but absent from the corpus - must not claim a match",
    },
    {
        "id": "T11", "query": "my payment keeps failing and i need this order placed today",
        "expect": {"payment_issue"},
        "urgency": (4, 5), "language": "en",
        "note": "In domain, blocked, deadline - the high-urgency case",
    },
]

#a handful of characters that only appear in Turkish, enough to tell the two apart here
TURKISH_MARKERS = set("çğıöşüÇĞİÖŞÜ")


def detect_language(text: str) -> str:
    return "tr" if TURKISH_MARKERS & set(text) else "en"


def score(case: dict, trace: dict) -> dict:
    """Turns one trace into pass/fail judgements against the case's expectations."""
    intents = trace["intents_hit"]
    top_intent = trace["retrieved"][0]["intent"]
    expected = case["expect"]
    low, high = case["urgency"]

    if expected:
        retrieval = "hit" if expected & set(intents) else "miss"
        ranking = "top" if top_intent in expected else "not top"
    else:
        #nothing to retrieve; the honest behaviour is to flag low confidence
        retrieval = "n/a"
        ranking = "n/a"

    return {
        "id": case["id"],
        "query": case["query"],
        "note": case["note"],
        "retrieval": retrieval,
        "ranking": ranking,
        "intents": intents,
        "best_cos": trace["best_cos"],
        "confidence": trace["confidence"],
        #an out-of-domain query should not be reported as a high-confidence match
        "confidence_ok": (trace["confidence"] != "high") if not expected else True,
        "urgency": trace["urgency"],
        "urgency_ok": trace["urgency"] is not None and low <= trace["urgency"] <= high,
        "urgency_band": f"{low}-{high}",
        "placeholder_leak": bool(re.search(r"\{\{.*?\}\}", trace["reply"])),
        "language": detect_language(trace["reply"]),
        "language_ok": detect_language(trace["reply"]) == case["language"],
        "seconds": trace["timing"]["total"],
        "reply": trace["reply"],
    }


def report(results: list) -> bool:
    """Prints the results table and the summary. Returns True if everything passed."""
    answerable = [r for r in results if r["retrieval"] != "n/a"]

    print(f"\n{'':<5}{'RETRIEVAL':<12}{'COS':<8}{'CONF':<9}{'URGENCY':<12}{'LANG':<7}{'PH':<5}QUERY")
    print("-" * 100)
    for r in results:
        retrieval = r["retrieval"] if r["retrieval"] == "n/a" else (
            "hit (top)" if r["ranking"] == "top" else f"{r['retrieval']}"
        )
        urgency = f"{r['urgency']} in {r['urgency_band']}" + ("" if r["urgency_ok"] else "  FAIL")
        print(
            f"{r['id']:<5}{retrieval:<12}{r['best_cos']:<8.3f}"
            f"{r['confidence'] + ('' if r['confidence_ok'] else '!'):<9}"
            f"{urgency:<12}{r['language'] + ('' if r['language_ok'] else '!'):<7}"
            f"{'LEAK' if r['placeholder_leak'] else 'ok':<5}{r['query'][:38]}"
        )

    hits = sum(r["retrieval"] == "hit" for r in answerable)
    tops = sum(r["ranking"] == "top" for r in answerable)
    urgency_ok = sum(r["urgency_ok"] for r in results)
    confidence_ok = sum(r["confidence_ok"] for r in results)
    language_ok = sum(r["language_ok"] for r in results)
    leaks = sum(r["placeholder_leak"] for r in results)

    in_domain = [r["best_cos"] for r in answerable]
    out_domain = [r["best_cos"] for r in results if r["retrieval"] == "n/a"]

    print(f"\n{'Retrieval hit rate':<26}{hits}/{len(answerable)}")
    print(f"{'Correct intent ranked #1':<26}{tops}/{len(answerable)}")
    print(f"{'Urgency in expected band':<26}{urgency_ok}/{len(results)}")
    print(f"{'Confidence honest':<26}{confidence_ok}/{len(results)}")
    print(f"{'Replied in query language':<26}{language_ok}/{len(results)}")
    print(f"{'Placeholder leaks':<26}{leaks}")
    if in_domain and out_domain:
        print(
            f"\nCosine separation: answerable {min(in_domain):.3f}-{max(in_domain):.3f}  |  "
            f"unanswerable {min(out_domain):.3f}-{max(out_domain):.3f}  |  "
            f"bands high ≥ {CONFIDENCE_HIGH}, low < {CONFIDENCE_LOW}"
        )
        if min(in_domain) <= max(out_domain):
            print("  WARNING: the two ranges overlap - the confidence threshold cannot separate them.")

    passed = (
        hits == len(answerable)
        and urgency_ok == len(results)
        and confidence_ok == len(results)
        and language_ok == len(results)
        and leaks == 0
    )
    print(f"\n{'PASS' if passed else 'FAIL'}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", metavar="PATH", help="also write raw results here")
    parser.add_argument("--only", nargs="+", metavar="ID", help="run only these case ids")
    parser.add_argument("--pause", type=float, default=4.0,
                        help="seconds between cases, to stay under the per-minute quota")
    args = parser.parse_args()

    index = faiss.read_index(INDEX_PATH)
    df = pd.read_csv(INDEXED_ROWS_PATH)

    cases = [c for c in CASES if not args.only or c["id"] in args.only]
    print(f"{index.ntotal} vectors, {df['intent'].nunique()} intents. Running {len(cases)} cases.")

    results = []
    for position, case in enumerate(cases):
        trace = trace_query(case["query"], index, df)
        results.append(score(case, trace))
        print(f"  {case['id']} done", end="\r")
        if position + 1 < len(cases):
            time.sleep(args.pause)

    passed = report(results)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        print(f"Raw results written to {args.json}")

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
