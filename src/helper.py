import logging
import os
import time
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

#the .env in this repo uses GEMINI_API_KEY, older docs/code used GOOGLE_API_KEY
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

#The SDK logs a one-off note about automatic function calling the first time
#generate_content is called. This project passes no tools, so it never applies - drop
#just that record rather than silencing the logger, which still carries real warnings.
logging.getLogger("google_genai.models").addFilter(
    lambda record: "automatic function calling" not in record.getMessage()
)

#google-genai replaced the retired google-generativeai package. One client is built on
#first use and reused; it holds the credentials and the HTTP session. Building it lazily
#rather than at import means the pure logic in this module can be unit tested, and CI
#run, without credentials - the same error still reaches anyone who runs the app.
_client = None


def get_client() -> genai.Client:
    """The shared Gemini client, built on first use."""
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "No API key found. Put GEMINI_API_KEY=<your key> in the .env file at "
                "the project root."
            )
        _client = genai.Client(api_key=API_KEY)
    return _client


#single source of truth for model names, so the index and the queries can never drift apart
EMBEDDING_MODEL = "models/gemini-embedding-001"
#free tier daily caps are what actually bite here: gemini-2.5-flash / 3.5-flash allow
#only 20 requests per DAY, while the flash-lite tier allows 500. For a support desk
#that answers many queries a day, flash-lite is the only usable choice on free tier.
LLM_MODEL = "models/gemini-3.5-flash-lite"


#single source of truth for model names, so the index and the queries can never drift apart
EMBEDDING_MODEL = "models/gemini-embedding-001"
#free tier daily caps are what actually bite here: gemini-2.5-flash / 3.5-flash allow
#only 20 requests per DAY, while the flash-lite tier allows 500. For a support desk
#that answers many queries a day, flash-lite is the only usable choice on free tier.
LLM_MODEL = "models/gemini-3.5-flash-lite"

class QuotaExhausted(RuntimeError):
    """The generation quota is used up - distinct from any other API failure."""


#gemini-embedding-001 defaults to 3072 dims; 768 keeps the FAISS index ~4x smaller
#with almost no retrieval quality loss. Truncated outputs are NOT unit length, so we
#normalise them ourselves before indexing/searching.
EMBEDDING_DIM = 768

#Free tier embedding quota, and every text inside a batch counts as its own request:
#  100 requests/minute, 30K tokens/minute, 1000 requests/DAY
#The daily cap is the hard one - it means at most 1000 rows can be indexed per day.
#The API also refuses more than 100 texts in a single batch outright, with a 400.
EMBED_BATCH_SIZE = 50
EMBED_MAX_BATCH = 100
EMBED_REQUESTS_PER_MINUTE = 90
EMBED_REQUESTS_PER_DAY = 1000


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalises rows so that L2 distance in FAISS ranks the same as cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


def embed_texts(
    texts: List[str],
    task_type: str = "retrieval_document",
    model: str = EMBEDDING_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embeds a list of texts with the Gemini embedding API.

    Sends texts in batches and paces itself against the free-tier quota, retrying with
    the delay the API asks for when a 429 comes back.

    Args:
        texts: the texts to embed.
        task_type: "retrieval_document" when indexing, "retrieval_query" when searching
            (case-insensitive; the API wants it upper-cased).
        model: the Gemini embedding model.
        batch_size: how many texts to send per request.
        show_progress: print progress while embedding.

    Returns:
        np.ndarray: normalised embeddings, shape (len(texts), EMBEDDING_DIM).
    """
    batch_size = min(batch_size, EMBED_MAX_BATCH)
    vectors: List[List[float]] = []
    #minimum seconds between requests to stay under the per-minute quota
    min_interval = (60.0 / EMBED_REQUESTS_PER_MINUTE) * batch_size
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]
        sent_at = time.time()

        #retry on quota errors; the API tells us how long to wait
        for attempt in range(6):
            try:
                res = get_client().models.embed_content(
                    model=model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type.upper(),
                        output_dimensionality=EMBEDDING_DIM,
                    ),
                )
                break
            except errors.APIError as exc:
                #429 / RESOURCE_EXHAUSTED is the only thing worth retrying
                if exc.code != 429:
                    raise
                wait = _retry_delay(str(exc), fallback=30 * (attempt + 1))
                if show_progress:
                    print(f"  quota hit, waiting {wait:.0f}s (attempt {attempt + 1}/6)")
                time.sleep(wait)
        else:
            raise RuntimeError("Embedding failed: quota exceeded after 6 retries.")

        vectors.extend(embedding.values for embedding in res.embeddings)

        if show_progress and total > batch_size:
            print(f"  embedded {min(start + batch_size, total)}/{total}")

        #pace the next request
        if start + batch_size < total:
            elapsed = time.time() - sent_at
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

    return _normalize(np.array(vectors, dtype="float32"))


def _retry_delay(message: str, fallback: float) -> float:
    """Pulls the retry delay the API suggested out of the error text."""
    marker = "Please retry in "
    if marker in message:
        try:
            return float(message.split(marker)[1].split("s")[0]) + 1.0
        except (IndexError, ValueError):
            pass
    return fallback


def create_embeddings(
    df: pd.DataFrame,
    column_name: str,
    model: str = EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Creates embeddings for one text column of a dataframe.

    Args:
        df: the dataframe holding the texts.
        column_name: the column to embed.
        model: the Gemini embedding model.

    Returns:
        np.ndarray: normalised embeddings, one row per dataframe row.
    """
    return embed_texts(df[column_name].astype(str).tolist(), task_type="retrieval_document", model=model)


def create_index(vectors: np.ndarray, index_file_path: str) -> faiss.Index:
    """
    This function creates a FAISS index, adds the provided vectors to the index, and saves it to a file.

    Args:
        vectors (np.ndarray): A NumPy array containing the vector embeddings.
        index_file_path (str): The path to save the FAISS index file.

    Returns:
        faiss.Index: The created FAISS index.
    """
    #make sure the folder exists before faiss tries to write into it
    folder = os.path.dirname(index_file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    #get the dimension of the vectors
    dimension = vectors.shape[1]
    #vectors are L2-normalised, so L2 distance here ranks identically to cosine distance
    index = faiss.IndexFlatL2(dimension)
    #add the vectors to the index
    index.add(vectors)
    #save the index to a file
    faiss.write_index(index, index_file_path)
    print(f"FAISS index is created with {index.ntotal} vectors and saved to {index_file_path}.")

    return index


def semantic_similarity(
    query: str,
    index: faiss.Index,
    model: str = EMBEDDING_MODEL,
    k: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the k most similar indexed entries for a query using Gemini embeddings.

    Args:
        query: the customer's inbound query.
        index: the FAISS index built from the dataset.
        model: the Gemini embedding model (must match the one used to build the index).
        k: how many neighbours to return.

    Returns:
        Tuple of (distances, indices), each of shape (1, k).
    """
    query_vector = embed_texts([query], task_type="retrieval_query", model=model, show_progress=False)
    return index.search(query_vector, k)


#The prompt, split at the two points where values are interpolated. Everything that
#displays or logs the prompt builds it from these parts, so what a trace shows can never
#drift from what was actually sent.
PROMPT_BEFORE = """You are drafting a reply for a human customer support agent to review.

CUSTOMER QUERY:
"""

PROMPT_MIDDLE = """

EXAMPLE REPLIES from our internal database, retrieved because they answer similar
queries. They are reference material, not a template to copy:
"""

PROMPT_AFTER = """

Return exactly three numbered items, nothing else.

1. Urgency, a single digit 1-5. Judge how fast a human must act, NOT how much work the
   request involves and NOT how many things were asked for. Use this scale:
   1 - general information, nothing is blocked, no action expected
   2 - a simple question the customer could self-serve
   3 - a routine action request with no deadline
   4 - the customer is blocked, or money or a deadline is at stake
   5 - repeated failure, an angry customer, or a live financial problem
   A customer calmly asking for three routine things is a 3, not a 5.

2. Category, one word: sales, product, operations, billing or account.

3. The drafted reply. Rules for it:
   - Write in the same language the customer used.
   - Synthesise the most complete and actionable answer the examples support. Do not
     copy the closest example verbatim - a short vague example is worse than a detailed
     one even when its wording is closer to the query.
   - Only promise steps and capabilities that appear in the examples. Never offer to
     check stock, look up an account, or perform an action the examples do not show.
   - If an example contains a {placeholder}, either fill it from the customer's own
     words or rewrite the sentence without it. Never leave {...} in the reply.
   - If the query covers several requests, address every one of them.
   - If the query is too vague to answer, ask one specific follow-up question instead.
"""


def build_prompt(query: str, responses: List[str]) -> str:
    """The exact prompt call_llm sends. Anything that displays it uses this."""
    return PROMPT_BEFORE + query + PROMPT_MIDDLE + str(responses) + PROMPT_AFTER


def prompt_parts(query: str, responses: List[str]) -> List[Dict[str, str]]:
    """
    The same prompt split into literal and interpolated spans, for viewers that want to
    highlight which parts came from the customer and which from the retrieved data.
    `kind` is "literal", "query", or "context".
    """
    return [
        {"kind": "literal", "text": PROMPT_BEFORE},
        {"kind": "query", "text": query},
        {"kind": "literal", "text": PROMPT_MIDDLE},
        {"kind": "context", "text": str(responses)},
        {"kind": "literal", "text": PROMPT_AFTER},
    ]


def call_llm(query: str, responses: List[str], model: str = LLM_MODEL) -> str:
    """
    Drafts an answer from the customer's query and the retrieved example replies.

    Args:
        query: the customer's inbound query.
        responses: similar responses retrieved from the internal dataset, used as context.
        model: the Gemini generative model.

    Returns:
        str: the model's answer.

    Raises:
        QuotaExhausted: the daily or per-minute generation quota is used up. Raised in
            place of the raw API error so callers can say something useful about it.
    """
    prompt = build_prompt(query, responses)

    try:
        response = get_client().models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )
    except errors.APIError as exc:
        if exc.code != 429:
            raise
        raise QuotaExhausted(
            f"The generation quota for {model.replace('models/', '')} is used up. "
            "The free tier allows 500 requests a day and 15 a minute - wait a minute and "
            "try again, or check https://ai.dev/rate-limit for where you stand."
        ) from exc

    return response.text
