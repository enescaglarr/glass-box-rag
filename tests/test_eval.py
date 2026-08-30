"""Unit tests for the scoring logic in eval.py - no API calls."""

import pytest

from eval import CASES, detect_language, score


def make_trace(intents, top_intent, best_cos, confidence, urgency, reply):
    return {
        "intents_hit": intents,
        "retrieved": [{"intent": top_intent}],
        "best_cos": best_cos,
        "confidence": confidence,
        "urgency": urgency,
        "reply": reply,
        "timing": {"total": 1.0},
    }


class TestDetectLanguage:
    @pytest.mark.parametrize("text,expected", [
        ("Please log in to your account.", "en"),
        ("Siparişinizi iptal etmek için giriş yapın.", "tr"),
        ("Hesabınıza giriş yapın", "tr"),
        ("", "en"),
    ])
    def test_detection(self, text, expected):
        assert detect_language(text) == expected


class TestScore:
    ANSWERABLE = {"id": "X", "query": "q", "note": "", "expect": {"cancel_order"},
                  "urgency": (2, 3), "language": "en"}
    UNANSWERABLE = {"id": "Y", "query": "q", "note": "", "expect": set(),
                    "urgency": (1, 3), "language": "en"}

    def test_hit_ranked_first(self):
        trace = make_trace(["cancel_order"], "cancel_order", 0.88, "high", 3, "Hello.")
        result = score(self.ANSWERABLE, trace)
        assert result["retrieval"] == "hit"
        assert result["ranking"] == "top"

    def test_hit_but_not_ranked_first(self):
        trace = make_trace(["change_order", "cancel_order"], "change_order",
                           0.83, "medium", 3, "Hello.")
        result = score(self.ANSWERABLE, trace)
        assert result["retrieval"] == "hit"
        assert result["ranking"] == "not top"

    def test_miss(self):
        trace = make_trace(["get_invoice"], "get_invoice", 0.79, "medium", 3, "Hello.")
        assert score(self.ANSWERABLE, trace)["retrieval"] == "miss"

    def test_unanswerable_query_is_not_scored_for_retrieval(self):
        trace = make_trace(["anything"], "anything", 0.75, "low", 2, "Hello.")
        result = score(self.UNANSWERABLE, trace)
        assert result["retrieval"] == "n/a"
        assert result["ranking"] == "n/a"

    def test_unanswerable_query_must_not_report_high_confidence(self):
        """The whole point of the confidence band: do not claim a match that isn't there."""
        honest = make_trace(["x"], "x", 0.75, "low", 2, "Hello.")
        dishonest = make_trace(["x"], "x", 0.75, "high", 2, "Hello.")
        assert score(self.UNANSWERABLE, honest)["confidence_ok"] is True
        assert score(self.UNANSWERABLE, dishonest)["confidence_ok"] is False

    def test_answerable_query_confidence_is_not_penalised(self):
        trace = make_trace(["cancel_order"], "cancel_order", 0.88, "high", 3, "Hello.")
        assert score(self.ANSWERABLE, trace)["confidence_ok"] is True

    @pytest.mark.parametrize("urgency,expected", [(1, False), (2, True), (3, True), (4, False)])
    def test_urgency_band(self, urgency, expected):
        trace = make_trace(["cancel_order"], "cancel_order", 0.88, "high", urgency, "Hi.")
        assert score(self.ANSWERABLE, trace)["urgency_ok"] is expected

    def test_missing_urgency_fails_rather_than_raising(self):
        trace = make_trace(["cancel_order"], "cancel_order", 0.88, "high", None, "Hi.")
        assert score(self.ANSWERABLE, trace)["urgency_ok"] is False

    def test_placeholder_leak_is_detected(self):
        leaked = make_trace(["cancel_order"], "cancel_order", 0.88, "high", 3,
                            "Cancel order {{Order Number}} from your account.")
        clean = make_trace(["cancel_order"], "cancel_order", 0.88, "high", 3,
                           "Cancel your order from your account.")
        assert score(self.ANSWERABLE, leaked)["placeholder_leak"] is True
        assert score(self.ANSWERABLE, clean)["placeholder_leak"] is False

    def test_language_mismatch_is_detected(self):
        turkish_reply = make_trace(["cancel_order"], "cancel_order", 0.88, "high", 3,
                                   "Siparişiniz iptal edildi.")
        assert score(self.ANSWERABLE, turkish_reply)["language_ok"] is False


class TestCases:
    def test_ids_are_unique(self):
        ids = [case["id"] for case in CASES]
        assert len(ids) == len(set(ids))

    def test_every_case_is_fully_specified(self):
        for case in CASES:
            assert case["query"].strip()
            assert isinstance(case["expect"], set)
            low, high = case["urgency"]
            assert 1 <= low <= high <= 5
            assert case["language"] in {"en", "tr"}
            assert case["note"].strip()

    def test_suite_covers_unanswerable_queries(self):
        #without these the suite cannot tell whether confidence means anything
        assert any(not case["expect"] for case in CASES)
