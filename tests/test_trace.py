"""Unit tests for the retrieval selection and output parsing in src.trace."""

import pytest

from src.trace import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    COSINE_TIE,
    confidence,
    parse_output,
    select_hits,
)


def candidate(cos, intent, response_length):
    return {"cos": cos, "intent": intent, "response": "x" * response_length}


class TestConfidence:
    @pytest.mark.parametrize("cos,expected", [
        (0.95, "high"), (CONFIDENCE_HIGH, "high"),
        (CONFIDENCE_HIGH - 0.001, "medium"), (CONFIDENCE_LOW, "medium"),
        (CONFIDENCE_LOW - 0.001, "low"), (0.0, "low"),
    ])
    def test_bands(self, cos, expected):
        assert confidence(cos) == expected

    def test_bands_are_ordered(self):
        assert CONFIDENCE_LOW < CONFIDENCE_HIGH


class TestSelectHits:
    def test_prefers_the_fuller_response_when_scores_tie(self):
        """
        The failure this exists for: a Turkish query retrieved three cancel_order rows
        0.002 apart, and the winner's stored reply was a 146-character non-answer while
        the other two were full step-by-step guides. Cosine ranks how similar the
        question is; it says nothing about whether the answer is any use.
        """
        candidates = [
            candidate(0.881, "cancel_order", 146),
            candidate(0.880, "cancel_order", 1153),
            candidate(0.879, "cancel_order", 981),
        ]
        assert len(select_hits(candidates, 1)[0]["response"]) == 1153

    def test_does_not_tie_break_across_a_real_gap(self):
        #0.05 apart is a genuine difference, not noise - the closer row must win
        candidates = [
            candidate(0.90, "cancel_order", 100),
            candidate(0.85, "cancel_order", 2000),
        ]
        assert select_hits(candidates, 1)[0]["cos"] == 0.90

    def test_tie_window_matches_the_constant(self):
        clearly_inside = [
            candidate(0.90, "a", 100),
            candidate(0.90 - COSINE_TIE / 4, "b", 900),
        ]
        assert len(select_hits(clearly_inside, 1)[0]["response"]) == 900

    def test_one_row_per_intent(self):
        """
        A compound query must not have every slot eaten by one intent - that is what
        happened to "cancel my order and also update my email and check my invoice".
        """
        candidates = [
            candidate(0.88, "cancel_order", 500),
            candidate(0.87, "cancel_order", 500),
            candidate(0.86, "cancel_order", 500),
            candidate(0.85, "edit_account", 500),
            candidate(0.84, "check_invoice", 500),
        ]
        intents = [hit["intent"] for hit in select_hits(candidates, 3)]
        assert len(set(intents)) == 3

    def test_falls_back_to_duplicates_when_intents_run_out(self):
        #a query genuinely about one intent should still get k rows
        candidates = [candidate(0.9 - i / 100, "cancel_order", 500) for i in range(4)]
        assert len(select_hits(candidates, 3)) == 3

    def test_returns_at_most_k(self):
        candidates = [candidate(0.9 - i / 100, f"intent_{i}", 500) for i in range(6)]
        assert len(select_hits(candidates, 3)) == 3

    def test_result_is_ordered_by_score(self):
        candidates = [candidate(0.9 - i / 100, f"intent_{i}", 500) for i in range(6)]
        scores = [hit["cos"] for hit in select_hits(candidates, 3)]
        assert scores == sorted(scores, reverse=True)

    def test_handles_fewer_candidates_than_k(self):
        assert len(select_hits([candidate(0.9, "a", 100)], 3)) == 1


class TestParseOutput:
    def test_splits_the_three_fields(self):
        parsed = parse_output("1. 4\n2. Operations\n3. Please log in and try again.")
        assert parsed["urgency"] == 4
        assert parsed["category_out"] == "Operations"
        assert parsed["reply"] == "Please log in and try again."

    def test_reply_keeps_its_own_numbered_list(self):
        """
        The parse markers are 1./2./3. at line start, and drafted replies now often
        contain a numbered list of their own. The real markers come first, so the reply
        must survive intact rather than being truncated at its own "1.".
        """
        raw = "1. 3\n2. operations\n3. To cancel:\n1. Log in.\n2. Open orders.\n3. Cancel."
        parsed = parse_output(raw)
        assert parsed["urgency"] == 3
        assert parsed["category_out"] == "operations"
        assert "Log in." in parsed["reply"]
        assert "Cancel." in parsed["reply"]

    def test_trailing_full_stop_is_stripped_from_the_category(self):
        assert parse_output("1. 2\n2. billing.\n3. Hi")["category_out"] == "billing"

    def test_multiline_reply_is_kept_whole(self):
        parsed = parse_output("1. 5\n2. billing\n3. First line.\n\nSecond line.")
        assert "First line." in parsed["reply"]
        assert "Second line." in parsed["reply"]

    def test_escaped_newlines_become_real_ones(self):
        #the model sometimes emits literal backslash-n inside its own text
        parsed = parse_output("1. 2\n2. sales\n3. One.\\nTwo.")
        assert "\\n" not in parsed["reply"]
        assert parsed["reply"].count("\n") == 1

    def test_missing_fields_do_not_raise(self):
        parsed = parse_output("something the model said instead")
        assert parsed["urgency"] is None
        assert parsed["category_out"] is None
        assert parsed["reply"] == "something the model said instead"

    def test_extra_whitespace_is_tolerated(self):
        parsed = parse_output("  1.  4 \n  2.  Operations \n  3.  Hello. ")
        assert parsed["urgency"] == 4
        assert parsed["category_out"] == "Operations"
        assert parsed["reply"] == "Hello."
