"""
Builds the RAG Trace Explorer - a local web page that records what the pipeline actually
did for a set of queries: the retrieved rows and their similarity scores, the prompt that
got assembled from them, and the model's output.

The page is written to report/ as plain static files and served over localhost.

Usage:
    python make_report.py --serve        # trace the queries, build, then serve on :8000
    python make_report.py                # trace and build only
    python make_report.py --cached       # rebuild the page from the last run, no API calls
    python make_report.py --serve --cached --port 8080
"""

import argparse
import http.server
import json
import os
import socketserver
import shutil
import webbrowser

import faiss
import pandas as pd

from src.helper import EMBEDDING_DIM, EMBEDDING_MODEL, LLM_MODEL
from src.trace import trace_query

INDEX_PATH = "vector_store/faiss_index.index"
INDEXED_ROWS_PATH = "vector_store/indexed_rows.csv"
TEMPLATE_DIR = "report_template"
OUT_DIR = "report"
TRACE_PATH = os.path.join(OUT_DIR, "trace.json")

#each query stresses a different part of the retriever
QUERIES = [
    ("i cant cancel my order, the website keeps erroring and i need my money back today",
     "Blocked and urgent"),
    ("how do i chnge the adress my stuff gets sent 2",
     "Typos and shorthand"),
    ("what payment methods do u accept",
     "Short and low-stakes"),
    ("this is the third time i've been charged twice, absolutely unacceptable",
     "Repeat billing complaint"),
]

def run_traces() -> dict:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(INDEXED_ROWS_PATH):
        raise SystemExit("Vector store not found. Build it first:\n    python build_index.py")

    index = faiss.read_index(INDEX_PATH)
    df = pd.read_csv(INDEXED_ROWS_PATH)
    print(f"Index: {index.ntotal} vectors, {index.d} dims. Tracing {len(QUERIES)} queries.")

    traces = []
    for query, note in QUERIES:
        #same function the Streamlit app calls, so the two can never disagree
        trace = trace_query(query, index, df)
        trace["note"] = note
        traces.append(trace)
        print(f"  traced: {query[:48]}...  "
              f"cos={trace['best_cos']:.3f} urgency={trace['urgency']} "
              f"({trace['timing']['total']:.2f}s)")

    return {
        "meta": {
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "dim": EMBEDDING_DIM,
            "index_vectors": int(index.ntotal),
            "dataset_rows": 26872,
            "indexed_rows": len(df),
            "intents": int(df["intent"].nunique()),
        },
        "traces": traces,
    }


def build_page(data: dict) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for asset in ("base.html", "style.css", "app.js"):
        source = os.path.join(TEMPLATE_DIR, asset)
        target = os.path.join(OUT_DIR, "index.html" if asset == "base.html" else asset)
        shutil.copyfile(source, target)

    with open(os.path.join(OUT_DIR, "data.js"), "w") as handle:
        handle.write("const DATA = " + json.dumps(data, indent=2) + ";\n")
    with open(TRACE_PATH, "w") as handle:
        json.dump(data, handle, indent=2)

    print(f"Report written to {OUT_DIR}/index.html")


def serve(port: int) -> None:
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=OUT_DIR, **kw)
    socketserver.TCPServer.allow_reuse_address = True
    url = f"http://localhost:{port}"
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"\nServing the report at {url}   (Ctrl+C to stop)")
        webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--serve", action="store_true", help="serve the report on localhost after building")
    parser.add_argument("--port", type=int, default=8000, help="port for --serve (default 8000)")
    parser.add_argument("--cached", action="store_true",
                        help="rebuild the page from the last recorded run instead of calling the API")
    args = parser.parse_args()

    if args.cached:
        if not os.path.exists(TRACE_PATH):
            raise SystemExit(f"No cached run at {TRACE_PATH}. Run without --cached first.")
        data = json.load(open(TRACE_PATH))
        print(f"Using the cached run from {TRACE_PATH} ({len(data['traces'])} queries).")
    else:
        data = run_traces()

    build_page(data)

    if args.serve:
        serve(args.port)


if __name__ == "__main__":
    main()
