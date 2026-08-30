import os
from datetime import datetime, timezone

import faiss
import pandas as pd
import streamlit as st

from src.helper import EMBEDDING_DIM, EMBEDDING_MODEL, LLM_MODEL
from src.trace import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    build_prompt,
    parse_output,
    trace_query,
)
from src.helper import QuotaExhausted, call_llm

INDEX_PATH = 'vector_store/faiss_index.index'
#build_index.py writes exactly the rows that went into the index; falling back to the
#full dataset would make the FAISS row ids point at the wrong records
INDEXED_ROWS_PATH = 'vector_store/indexed_rows.csv'
#every accepted draft is appended here, so the review loop leaves a record
ACCEPTED_PATH = 'accepted_replies.csv'
ACCEPTED_COLUMNS = [
    "accepted_at", "query", "urgency", "category", "confidence",
    "best_cos", "intents", "model", "reply",
]

st.set_page_config(page_title="AI Assisted Customer Support", page_icon="🎧", layout="wide")

#Only spacing and the reply card - everything else is stock Streamlit so the app keeps
#working in both light and dark themes without a second palette to maintain.
st.html("""
<style>
  .block-container {padding-top: 2.5rem; max-width: 1180px;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  /* the drafted reply sits in the first bordered container of the left column */
  div[data-testid="stVerticalBlockBorderWrapper"] {line-height: 1.6;}
  .stTabs [data-baseweb="tab"] {font-size: .92rem;}
</style>
""")

SAMPLES = [
    "I need to change the shipping address on my order",
    "helo i cnt log in to my acount pls help",
    "WHY HAVE I STILL NOT RECEIVED MY REFUND?!?!",
    "siparişimi iptal etmek istiyorum",
    "do you have this in size medium",
]

#colour, icon and one line of guidance for the reviewer, per retrieval confidence band
CONFIDENCE_STYLE = {
    "high": ("green", "✓", "The dataset answers this kind of query directly."),
    "medium": ("orange", "!", "Partial match — check the retrieved rows before sending."),
    "low": ("red", "✕", "No good answer in the dataset. Handle this one yourself."),
}
URGENCY_LABEL = {1: "Informational", 2: "Low", 3: "Routine", 4: "Blocked", 5: "Escalate"}


def load_accepted() -> pd.DataFrame:
    """Everything accepted so far, or an empty frame with the right columns."""
    if os.path.exists(ACCEPTED_PATH):
        return pd.read_csv(ACCEPTED_PATH)
    return pd.DataFrame(columns=ACCEPTED_COLUMNS)


def append_accepted(trace) -> None:
    """Appends one accepted draft to the log, creating the file on first use."""
    row = pd.DataFrame([{
        "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": trace["query"],
        "urgency": trace["urgency"],
        "category": trace["category_out"],
        "confidence": trace["confidence"],
        "best_cos": trace["best_cos"],
        "intents": " ".join(trace["intents_hit"]),
        "model": trace.get("llm_model", ""),
        "reply": trace["reply"],
    }])
    row.to_csv(
        ACCEPTED_PATH,
        mode="a" if os.path.exists(ACCEPTED_PATH) else "w",
        header=not os.path.exists(ACCEPTED_PATH),
        index=False,
    )


@st.cache_resource
def load_store():
    """Loads the FAISS index and its rows once per server process."""
    return faiss.read_index(INDEX_PATH), pd.read_csv(INDEXED_ROWS_PATH)


def hits_table(trace):
    """Every neighbour that was fetched, sent and dropped alike, as one sortable table."""
    rows = [
        {
            "Sent": True,
            "Match": hit["cos"],
            "Intent": hit["intent"],
            "Closest question in the dataset": hit["instruction"],
            "Reply length": len(hit["response"]),
        }
        for hit in trace["retrieved"]
    ] + [
        {
            "Sent": False,
            "Match": candidate["cos"],
            "Intent": candidate["intent"],
            "Closest question in the dataset": candidate["instruction"],
            "Reply length": len(candidate["response"]),
        }
        for candidate in trace["dropped"]
    ]
    return pd.DataFrame(rows).sort_values("Match", ascending=False)


def render_answer(trace):
    """Verdict chips and the drafted reply - what an agent acts on."""
    colour, icon, guidance = CONFIDENCE_STYLE[trace["confidence"]]
    urgency = trace["urgency"] or 0

    chips = st.columns([1.4, 1, 1.2])
    chips[0].markdown(
        f":{colour}-badge[{icon} Retrieval {trace['confidence']}] &nbsp; "
        f"`{trace['best_cos']:.3f}`"
    )
    chips[1].markdown(
        f":{'red' if urgency >= 5 else 'orange' if urgency >= 4 else 'blue'}-badge"
        f"[Urgency {urgency}/5 · {URGENCY_LABEL.get(urgency, '—')}]"
    )
    chips[2].markdown(f":violet-badge[{(trace['category_out'] or '—').title()}]")
    st.caption(guidance)

    st.markdown("##### Drafted reply")
    #a native bordered container, not a raw <div> - the reply often contains a numbered
    #list, and markdown inside hand-written HTML does not get rendered
    with st.container(border=True):
        st.markdown(trace["reply"])


def render_trace(trace):
    """The four pipeline stages, one per tab instead of four stacked expanders."""
    timing = trace["timing"]
    retrieval, prompt_tab, output_tab, vector_tab = st.tabs([
        f"Retrieval · {len(trace['retrieved'])} of {len(trace['retrieved']) + len(trace['dropped'])} sent",
        f"Prompt · {trace['prompt_chars']:,} chars",
        "Model output",
        f"Vector & timing · {timing['total']:.2f}s",
    ])

    with retrieval:
        st.dataframe(
            hits_table(trace),
            hide_index=True,
            width="stretch",
            column_config={
                "Sent": st.column_config.CheckboxColumn(
                    "Sent", width="small",
                    help="Six neighbours are fetched, then narrowed: one row per intent, "
                         "and where scores tie the fuller reply wins.",
                ),
                "Match": st.column_config.ProgressColumn(
                    "Match", format="%.3f", min_value=0.6, max_value=1.0, width="small",
                    help=f"Cosine similarity. High ≥ {CONFIDENCE_HIGH}, low < {CONFIDENCE_LOW}.",
                ),
                "Intent": st.column_config.TextColumn("Intent", width="medium"),
                "Reply length": st.column_config.NumberColumn(
                    "Chars", format="%d", width="small",
                    help="Length of the stored reply. Only the reply text is sent to the model.",
                ),
            },
        )

        labels = {
            f"{h['cos']:.3f} · {h['intent']} · {h['instruction'][:52]}": h
            for h in trace["retrieved"]
        }
        chosen = st.selectbox("Read the full stored reply for", list(labels), key="hit_pick")
        if chosen:
            st.info(labels[chosen]["response"])

    with prompt_tab:
        st.caption(
            "Everything after the instructions is retrieved data, injected as a Python list."
        )
        st.code(trace["prompt"], language=None, wrap_lines=True)

    with output_tab:
        st.caption(f"{LLM_MODEL.replace('models/', '')} · temperature 0 · parsed into the three fields above")
        st.code(trace["output"], language=None, wrap_lines=True)

    with vector_tab:
        stages = st.columns(4)
        stages[0].metric("Embed", f"{timing['embed']:.2f}s")
        stages[1].metric("Search", f"{timing['search'] * 1000:.1f}ms")
        stages[2].metric("Generate", f"{timing['llm']:.2f}s")
        stages[3].metric("Total", f"{timing['total']:.2f}s")
        st.caption(
            f"Query embedding — first 8 of {EMBEDDING_DIM} dimensions, "
            f"L2 norm {trace['vector_norm']:.4f} after normalising."
        )
        st.code("  ".join(f"{v:+.4f}" for v in trace["query_vector_head"]) + "  …", language=None)


#--- page ---------------------------------------------------------------------------

if not os.path.exists(INDEX_PATH) or not os.path.exists(INDEXED_ROWS_PATH):
    st.title("AI Assisted Customer Support")
    st.error("No vector store yet. Build it with `python build_index.py`, then reload.")
    st.stop()

index, df = load_store()

with st.sidebar:
    st.subheader("System")
    st.caption("Retrieval")
    st.code(f"{EMBEDDING_MODEL.replace('models/', '')}\n{EMBEDDING_DIM}d · FAISS IndexFlatL2", language=None)
    st.caption("Generation")
    st.code(LLM_MODEL.replace("models/", ""), language=None)
    st.metric("Indexed rows", f"{index.ntotal:,}")
    st.metric("Intents covered", f"{df['intent'].nunique()} of 27")

    st.divider()
    accepted = load_accepted()
    st.subheader(f"Accepted · {len(accepted)}")
    if accepted.empty:
        st.caption("Nothing accepted yet. Approved drafts are logged here.")
    else:
        st.caption(f"Last: {accepted.iloc[-1]['query'][:44]}")
        st.download_button(
            "Download CSV",
            accepted.to_csv(index=False).encode("utf-8"),
            file_name=ACCEPTED_PATH,
            mime="text/csv",
            width="stretch",
        )
        with st.expander("Review the log"):
            st.dataframe(
                accepted[["accepted_at", "query", "urgency", "category", "confidence"]],
                hide_index=True,
                width="stretch",
            )

    st.divider()
    st.subheader("Try one")
    for number, sample in enumerate(SAMPLES):
        if st.button(sample, key=f"sample_{number}", width="stretch"):
            st.session_state["query"] = sample
            st.session_state.pop("trace", None)
            st.rerun()

st.title("AI Assisted Customer Support")
st.caption("Retrieves past support replies for an inbound query, then drafts an answer an agent reviews.")

with st.form("ask", border=False):
    field, submit = st.columns([5, 1], vertical_alignment="bottom")
    query = field.text_input("Inbound query", key="query", placeholder="What did the customer write?")
    asked = submit.form_submit_button("Answer", type="primary", width="stretch")

if asked:
    if not query:
        st.warning("Type a query first.")
    else:
        try:
            with st.spinner("Embedding, searching, generating…"):
                st.session_state["trace"] = trace_query(query, index, df)
        except QuotaExhausted as exhausted:
            st.error(f"**Out of quota.** {exhausted}")
        except Exception as failure:
            #a live demo should explain itself rather than show a traceback
            st.error(f"**{type(failure).__name__}** — {failure}")
            st.caption(
                "Usually a rate limit or a dropped connection. Wait a moment and try again; "
                "if it persists, check your key in `.env` and your quota at ai.dev/rate-limit."
            )

if "trace" in st.session_state:
    trace = st.session_state["trace"]

    st.divider()
    answer, evidence = st.columns([1, 1.25], gap="large")

    with answer:
        render_answer(trace)

        with st.expander("Send it back for a rewrite"):
            feedback = st.text_area(
                "What should change?",
                key="feedback",
                placeholder="shorter, and don't ask them to phone in",
                label_visibility="collapsed",
            )
            accept, regenerate = st.columns(2)
            if accept.button("Accept", width="stretch"):
                append_accepted(trace)
                st.success(f"Saved to `{ACCEPTED_PATH}`.")
            if regenerate.button("Rewrite", type="primary", width="stretch"):
                if not feedback:
                    st.warning("Say what should change first.")
                else:
                    responses = [hit["response"] for hit in trace["retrieved"]]
                    #only point 3 gets rewritten; the urgency and category judgements stand
                    rewrite = (
                        f"Regenerate third point of this response: {trace['output']}.\n\n"
                        "You must only regenerate third point according to the feedback later. "
                        "Do not change 1st and 2nd point at any cost but always have them in the "
                        f"final output.\nFeedback: {feedback}"
                    )
                    try:
                        with st.spinner("Rewriting…"):
                            answer = call_llm(rewrite, responses)
                    except QuotaExhausted as exhausted:
                        st.error(f"**Out of quota.** {exhausted}")
                        answer = None
                    except Exception as failure:
                        st.error(f"**{type(failure).__name__}** — {failure}")
                        answer = None

                    if answer is not None:
                        #keep the same trace, swap in the new answer so the retrieval detail still applies
                        trace = dict(trace)
                        trace["output"] = answer.strip()
                        trace.update(parse_output(answer))
                        trace["prompt"] = build_prompt(rewrite, responses)
                        trace["prompt_chars"] = len(trace["prompt"])
                        st.session_state["trace"] = trace
                        st.rerun()

    with evidence:
        st.markdown("##### How this answer was produced")
        render_trace(trace)
