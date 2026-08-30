"""Unit tests for the pure logic in src.helper - no API calls, no credentials needed."""

import numpy as np
import pytest

from src.helper import (
    EMBED_BATCH_SIZE,
    EMBED_MAX_BATCH,
    PROMPT_AFTER,
    PROMPT_BEFORE,
    PROMPT_MIDDLE,
    _normalize,
    build_prompt,
    prompt_parts,
)


class TestNormalize:
    def test_rows_become_unit_length(self):
        vectors = np.array([[3.0, 4.0], [1.0, 0.0], [-2.0, -2.0]], dtype="float32")
        assert np.allclose(np.linalg.norm(_normalize(vectors), axis=1), 1.0)

    def test_direction_is_preserved(self):
        vectors = np.array([[3.0, 4.0]], dtype="float32")
        assert np.allclose(_normalize(vectors), [[0.6, 0.8]])

    def test_zero_vector_does_not_divide_by_zero(self):
        #Gemini should never return one, but a NaN row would silently poison the index
        result = _normalize(np.array([[0.0, 0.0], [1.0, 0.0]], dtype="float32"))
        assert not np.isnan(result).any()
        assert np.allclose(result[0], [0.0, 0.0])

    def test_output_is_float32_for_faiss(self):
        #faiss rejects float64 input
        assert _normalize(np.array([[1.0, 2.0]], dtype="float64")).dtype == np.float32


class TestPrompt:
    """
    The parts and the whole must never disagree.

    They did once: the prompt was rewritten in helper.py while a second copy lived in
    trace.py, and the app's prompt panel spent a while displaying a prompt that was not
    the one being sent. These tests exist so that cannot recur silently.
    """

    QUERY = "cancel my order"
    RESPONSES = ["Log in and cancel from the orders page.", "Contact support."]

    def test_parts_join_to_the_whole(self):
        joined = "".join(part["text"] for part in prompt_parts(self.QUERY, self.RESPONSES))
        assert joined == build_prompt(self.QUERY, self.RESPONSES)

    def test_query_and_context_are_interpolated(self):
        prompt = build_prompt(self.QUERY, self.RESPONSES)
        assert self.QUERY in prompt
        assert str(self.RESPONSES) in prompt

    def test_every_part_is_labelled(self):
        kinds = [part["kind"] for part in prompt_parts(self.QUERY, self.RESPONSES)]
        assert set(kinds) == {"literal", "query", "context"}
        assert kinds.count("query") == 1
        assert kinds.count("context") == 1

    def test_literal_parts_carry_no_interpolated_values(self):
        for part in prompt_parts(self.QUERY, self.RESPONSES):
            if part["kind"] == "literal":
                assert self.QUERY not in part["text"]

    def test_prompt_states_the_urgency_failure_mode(self):
        #the calibration fix this scale exists for: urgency tracked workload, not urgency
        assert "NOT how much work" in PROMPT_AFTER
        assert "three routine things is a 3" in PROMPT_AFTER

    def test_prompt_forbids_leaving_placeholders(self):
        assert "Never leave" in PROMPT_AFTER

    def test_prompt_asks_for_the_customers_language(self):
        assert "same language the customer used" in PROMPT_AFTER

    @pytest.mark.parametrize("part", [PROMPT_BEFORE, PROMPT_MIDDLE, PROMPT_AFTER])
    def test_parts_are_non_empty(self, part):
        assert part.strip()


class TestQuotaConstants:
    def test_batch_size_stays_under_the_api_hard_limit(self):
        #the API rejects a batch of more than 100 outright, with a 400
        assert EMBED_BATCH_SIZE <= EMBED_MAX_BATCH == 100
