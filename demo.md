# Pipeline Test Report — Before and After

Eight queries run through the live pipeline, first against the original code and then
against the fixed version. Every number here is recorded output from an actual run, not
an estimate.

**Setup, identical in both runs**

| | |
|---|---|
| Retrieval | `gemini-embedding-001`, 768 dims, L2-normalised |
| Index | FAISS `IndexFlatL2`, 270 vectors, 27 intents, 10 rows per intent |
| Generation | `gemini-3.5-flash-lite`, temperature 0 |
| Corpus | Bitext customer support dataset, 26,872 rows sampled down to 270 |

The index was **not** rebuilt between runs. Every improvement below comes from the
retrieval selection logic and the prompt, not from better embeddings.

---

## The eight queries

Each was chosen to stress a different part of the pipeline.

| | Query | What it tests |
|---|---|---|
| T1 | `I need to change the shipping address on my order` | Clean in-domain baseline |
| T2 | `cancel order` | Keyword mode, no sentence structure |
| T3 | `helo i cnt log in to my acount pls help` | Heavy typos |
| T4 | `WHY HAVE I STILL NOT RECEIVED MY REFUND?!?! this is ridiculous` | Anger, caps, real urgency |
| T5 | `do you have this in size medium` | Out of domain — no such intent exists |
| T6 | `siparişimi iptal etmek istiyorum` | Turkish, cross-lingual retrieval |
| T7 | `it doesn't work` | Too vague to answer |
| T8 | `i want to cancel my order and also update my email and check my invoice` | Three intents in one message |

---

## Summary

| | Urgency | | Retrieval | | Reply | |
|---|---|---|---|---|---|---|
| | before | after | intents before | after | before | after |
| T1 | 4 | **3** | 2 | 2 | verbatim copy | synthesised |
| T2 | 4 | **3** | 1 | 1 | verbatim copy, placeholders leaked | synthesised, clean |
| T3 | 4 | 4 | 2 | **3** | synthesised | synthesised |
| T4 | 5 | 5 | 2 | 2 | synthesised | synthesised |
| T5 | 3 | **2** | 2 | **3** | promised a stock check | no false promise |
| T6 | 4 | **3** | 1 | 1 | **English reply, 146-char stub** | **Turkish, full instructions** |
| T7 | 4 | 4 | 2 | **3** | synthesised | synthesised |
| T8 | **5** | **3** | 2 | 2 | acknowledged only | steps for the main request |

Placeholders (`{{Order Number}}` and friends) leaking into the drafted reply: **2 of 8
before, 0 of 8 after.**

---

## What was wrong, and what changed

### 1. Cosine ranked the question, never the answer

The sharpest failure. In **T6** the three retrieved rows were separated by 0.002 cosine —
statistical noise — but their responses were worth wildly different amounts:

| | cosine | response | content |
|---|---|---|---|
| #1 *chosen* | 0.8813 | **146 chars** | "I'm here to provide guidance and support" — no actual steps |
| #2 | 0.8794 | 1153 chars | full step-by-step cancellation guide |
| #3 | 0.8738 | 981 chars | full step-by-step cancellation guide |

The stub won because its *instruction* was phrased closest to the query. Its *response*
being useless was invisible to the pipeline. The same query asked in English (T2) landed
on a useful row instead — output quality was decided by 0.002 of noise.

**Fix.** Fetch 6 neighbours instead of 3, and where scores tie within 0.01, prefer the
fuller response ([`select_hits`](src/trace.py)).

**Result.** T6 now keeps the 1153 / 1069 / 1082-character guides and drops the 146-char
stub. T2 and T8 drop the same stub. T6's reply went from a content-free acknowledgement
to a numbered cancellation procedure.

### 2. Urgency measured workload, not urgency

Before, the score tracked how much *work* a request implied rather than how fast a human
had to act. **T8** — a calm customer asking for three routine things — scored **5/5**, the
same as T4's angry customer chasing a missing refund. Routine requests sat at a floor of
4; the value 1 never appeared.

**Fix.** The prompt now anchors every level with a description and states the failure mode
explicitly: *"Judge how fast a human must act, NOT how much work the request involves and
NOT how many things were asked for. A customer calmly asking for three routine things is
a 3, not a 5."*

**Result.** T8 dropped 5 → 3. Routine actions (T1, T2, T6) dropped 4 → 3. Informational
queries (T5) dropped to 2. T4, the genuinely urgent one, stayed at 5. The scale now
separates *angry / blocked* from *routine* instead of compressing everything into 4–5.

### 3. Replies ignored the customer's language

**T6** was written in Turkish and answered in English. Nothing in the prompt asked the
model to mirror the query language, so it mirrored the retrieved rows instead.

**Fix.** One line: *"Write in the same language the customer used."*

**Result.** T6 now answers in Turkish, with the retrieved English procedure translated
into it — cross-lingual retrieval feeding a same-language reply.

### 4. The model copied instead of synthesising

At high similarity the drafted reply was a **character-for-character copy** of the nearest
row (T1: 397 chars, T2: 981 chars). Generation added nothing but latency and quota. Worse,
copying carried the dataset's unfilled placeholders straight through to the customer:

```
{{Order Number}}  {{Online Company Portal Info}}
{{Online Order Interaction}} ×4  {{Customer Support Hours}}  {{Website URL}}
```

`{{Online Order Interaction}}` appears four times in one reply meaning three different
things — the orders page, the cancel button, the order list. Even a CRM could not fill it
correctly; the information is lost in the source data.

**Fix.** The prompt reframes the retrieved rows as *"reference material, not a template to
copy"*, and adds: *"Never leave `{{...}}` in the reply"* — fill it from the customer's own
words or rewrite the sentence without it.

**Result.** No verbatim copies in any of the eight. Zero placeholder leaks.

### 5. Compound queries had all three slots eaten by one intent

**T8** asked for three things. All three retrieved rows came back `ORDER` — nothing about
changing an email or fetching an invoice. Two thirds of the request had no context at all.

**Fix.** Deduplicate by intent: at most one row per intent survives into the prompt.

**Result.** Partial. T8 now retrieves two distinct intents instead of one, and T3, T5 and
T7 each retrieve three. But T8's reply still only gives procedural steps for the
cancellation — see the honest assessment below.

### 6. Nothing told the reviewer when retrieval had failed

The best-match cosine cleanly separates queries the dataset can answer from queries it
cannot, but that signal was never surfaced:

```
in domain    0.832 ─ 0.886          T7 (vague) 0.801          out of domain 0.714 ─ 0.765
```

**Fix.** A confidence band above every answer — high ≥ 0.84, low < 0.78 — with an
instruction for the reviewer rather than just a number. The panel also now lists the
neighbours that were fetched but not sent, with their scores and response lengths.

**Result.** T5 (`size medium`) flags **low** and tells the agent to handle it themselves.
T7 and T8 flag **medium**. The other five flag **high**.

---

## What did not get fixed

**T8 still under-answers.** The reply acknowledges all three requests but gives real steps
only for the cancellation. Intent deduplication cannot help here: the query vector is
dominated by "cancel my order", so `edit_account` and `check_invoice` rows never make the
top 6 at all. Properly fixing this needs the query split into separate intents and
retrieved for separately — a different piece of work.

**T5 still over-promises, mildly.** The old reply said *"we will gladly check our stock"* —
a capability that does not exist. The new one says the customer can *"browse our website or
catalog"* to check availability. Better grounded, but browsing a catalogue is not really
how you check whether a size is in stock. The dataset simply has no intent for this and
the model fills the gap with something plausible.

**T7 urgency is still a guess.** `it doesn't work` scores 4. There is genuinely no way to
assess urgency from that string, so any number is arbitrary — but 4 reads as more
confident than the situation warrants.

**Output parsing stays fragile.** The three answers are parsed out of the raw text by
matching `1.`, `2.` and `3.` at line starts. When the drafted reply itself contains a
numbered list — as T2's now does — the parse still works because the real markers come
first, but the format is doing more work than it should.

---

## What consistently worked, in both runs

**Retrieval is the strong part of this system.** It handled typos (`helo i cnt log in to my
acount`, 0.842), keyword mode (`cancel order`, 0.866), shouting and punctuation
(`WHY HAVE I STILL NOT...?!?!`, 0.846), and Turkish (`siparişimi iptal etmek istiyorum`,
**0.880** — higher than the same request in English). All on 270 vectors, 1% of the dataset.

**The follow-up instruction is the single most valuable line in the prompt.** *"If the query
is too vague to answer, ask one specific follow-up question instead"* fired correctly in T3,
T5 and T7 in both runs. In T7 the retrieved rows were about payment failures and invoices,
and the model refused to be dragged there — it asked what the customer was actually trying
to do. That one line is what keeps out-of-domain queries from becoming confident fiction.

---

## The changes

| File | Change |
|---|---|
| [`src/helper.py`](src/helper.py) | Prompt rewritten: anchored 1–5 urgency scale with the workload failure mode called out, mirror the customer's language, synthesise rather than copy, never emit `{{placeholders}}`, never promise capabilities absent from the examples, address every request in a compound query |
| [`src/trace.py`](src/trace.py) | `select_hits()` — over-fetch 6, one row per intent, tie-break within 0.01 cosine toward the fuller response. `confidence()` — high / medium / low bands. Trace now carries `confidence` and the `dropped` candidates |
| [`demo.py`](demo.py) | Confidence banner above the answer with a reviewer instruction; the retrieval panel lists what was fetched and discarded |

The index was untouched. Rebuilding it over `instruction + response` instead of
`instruction` alone was considered as a deeper fix for problem 1 and deliberately not
done — it changes the retrieval space, costs a day of embedding quota, and the tie-break
already resolves the observed failure without that risk.
