"""Tests for GammaGuardrail: cascade routing correctness and
a real (non-mocked) calibration/scoring integration check.

Cascade-boundary tests monkeypatch survival_score() to return controlled
values — the reject/approve/escalate classification logic is what's under
test, not the embedding model's actual output, which would make boundary
tests flaky. The integration test at the bottom exercises the real
encoder + real scipy Gamma fit end-to-end.
"""
import numpy as np
import pytest

from evaluation.gamma_guardrail import GammaGuardrail

SYNTHETIC_REFERENCE_CORPUS = [
    "Retrieval-augmented generation grounds language model outputs in retrieved documents.",
    "The transformer architecture relies on self-attention to model long-range dependencies.",
    "Cross-validation is used to estimate the generalization error of a machine learning model.",
    "Gradient descent iteratively updates model parameters to minimize a loss function.",
    "Convolutional neural networks apply learned filters across spatial input dimensions.",
    "Reinforcement learning agents optimize a policy to maximize cumulative reward.",
    "Bayesian inference updates a posterior distribution given observed data and a prior.",
    "The attention mechanism computes a weighted sum over value vectors using query-key similarity.",
    "Regularization techniques such as dropout reduce overfitting in deep neural networks.",
    "Embedding models map discrete tokens into continuous vector representations.",
    "Beam search explores multiple candidate sequences during autoregressive decoding.",
    "Batch normalization stabilizes training by normalizing layer activations.",
    "Transfer learning reuses representations learned on a source task for a target task.",
    "The encoder-decoder architecture is common in sequence-to-sequence modeling.",
    "Hyperparameter tuning searches over configurations to optimize validation performance.",
    "Knowledge distillation trains a smaller student model to mimic a larger teacher model.",
    "Data augmentation increases training set diversity through label-preserving transformations.",
    "Positional encodings inject sequence order information into transformer inputs.",
    "The loss landscape of deep networks is highly non-convex with many local minima.",
    "Federated learning trains models across decentralized data without centralizing it.",
]


class TestCalibrationValidation:
    def test_calibrate_requires_at_least_two_texts(self):
        guardrail = GammaGuardrail()
        with pytest.raises(ValueError):
            guardrail.calibrate(["only one sentence"])

    def test_survival_score_before_calibration_raises(self):
        guardrail = GammaGuardrail()
        with pytest.raises(RuntimeError):
            guardrail.survival_score(["some text"])

    def test_is_calibrated_flag(self):
        guardrail = GammaGuardrail()
        assert guardrail.is_calibrated is False


class TestCascadeRoutingCorrectness:
    """score_and_route's reject/approve/escalate boundaries, isolated from
    the embedding model via a monkeypatched survival_score."""

    @pytest.fixture
    def guardrail(self, monkeypatch):
        guardrail = GammaGuardrail.__new__(GammaGuardrail)  # skip __init__, no encoder needed
        guardrail._gamma_params = (1.0, 0.0, 1.0)  # pretend calibrated

        def fake_survival_score(texts):
            # deterministic per-sentence score keyed on a marker token
            mapping = {"REJECT": 0.01, "ESCALATE": 0.15, "APPROVE": 0.9}
            return np.array([mapping[t.split()[0]] for t in texts])

        monkeypatch.setattr(guardrail, "survival_score", fake_survival_score)
        return guardrail

    def test_below_certain_wrong_is_reject(self, guardrail):
        decisions, _ = guardrail.score_and_route("REJECT this sentence.", sf_threshold=0.15)
        assert decisions == ["reject"]

    def test_above_certain_right_is_approve(self, guardrail):
        decisions, _ = guardrail.score_and_route("APPROVE this sentence.", sf_threshold=0.15)
        assert decisions == ["approve"]

    def test_between_bounds_is_escalate(self, guardrail):
        decisions, _ = guardrail.score_and_route("ESCALATE this sentence.", sf_threshold=0.15)
        assert decisions == ["escalate"]

    def test_boundary_values_are_exclusive_at_certain_wrong(self):
        """SF exactly at CERTAIN_WRONG (0.05) should NOT be "reject" —
        the doc's condition is strictly `sf < CERTAIN_WRONG`."""
        guardrail = GammaGuardrail.__new__(GammaGuardrail)
        guardrail._gamma_params = (1.0, 0.0, 1.0)
        guardrail.survival_score = lambda texts: np.array([GammaGuardrail.CERTAIN_WRONG])
        decisions, _ = guardrail.score_and_route("boundary sentence.", sf_threshold=0.15)
        assert decisions == ["escalate"]

    def test_boundary_values_are_exclusive_at_certain_right(self):
        """SF exactly at CERTAIN_RIGHT (0.25) should NOT be "approve" —
        the doc's condition is strictly `sf > CERTAIN_RIGHT`."""
        guardrail = GammaGuardrail.__new__(GammaGuardrail)
        guardrail._gamma_params = (1.0, 0.0, 1.0)
        guardrail.survival_score = lambda texts: np.array([GammaGuardrail.CERTAIN_RIGHT])
        decisions, _ = guardrail.score_and_route("boundary sentence.", sf_threshold=0.15)
        assert decisions == ["escalate"]

    def test_mixed_draft_routes_each_sentence_independently(self, guardrail):
        draft = "REJECT this one. ESCALATE this one. APPROVE this one."
        decisions, mean_sf = guardrail.score_and_route(draft, sf_threshold=0.15)
        assert decisions == ["reject", "escalate", "approve"]
        assert mean_sf == pytest.approx((0.01 + 0.15 + 0.9) / 3)

    def test_empty_draft_returns_empty_decisions(self, guardrail):
        decisions, mean_sf = guardrail.score_and_route("", sf_threshold=0.15)
        assert decisions == []
        assert mean_sf == 0.0

    def test_sf_threshold_does_not_move_cascade_boundaries(self, guardrail):
        """Per the module docstring: sf_threshold is accepted for
        signature parity with filter_chunks but the cascade always uses
        the fixed CERTAIN_WRONG/CERTAIN_RIGHT constants."""
        decisions_low, _ = guardrail.score_and_route("ESCALATE this.", sf_threshold=0.0)
        decisions_high, _ = guardrail.score_and_route("ESCALATE this.", sf_threshold=0.99)
        assert decisions_low == decisions_high == ["escalate"]


class TestFilterChunks:
    @pytest.fixture
    def guardrail(self, monkeypatch):
        guardrail = GammaGuardrail.__new__(GammaGuardrail)
        guardrail._gamma_params = (1.0, 0.0, 1.0)
        monkeypatch.setattr(
            guardrail,
            "survival_score",
            lambda texts: np.array([0.9, 0.1, 0.5]),
        )
        return guardrail

    def test_keeps_only_chunks_above_threshold(self, guardrail):
        chunks = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        verified, scores = guardrail.filter_chunks(chunks, sf_threshold=0.3)
        assert verified == [{"content": "a"}, {"content": "c"}]
        assert scores == pytest.approx([0.9, 0.5])

    def test_empty_chunks_returns_empty(self, guardrail):
        verified, scores = guardrail.filter_chunks([], sf_threshold=0.3)
        assert verified == []
        assert scores == []

    def test_threshold_of_zero_keeps_everything(self, guardrail):
        chunks = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        verified, _ = guardrail.filter_chunks(chunks, sf_threshold=0.0)
        assert len(verified) == 3

    def test_threshold_of_one_keeps_nothing(self, guardrail):
        chunks = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        verified, _ = guardrail.filter_chunks(chunks, sf_threshold=1.0)
        assert verified == []


class TestRealIntegration:
    """Real encoder + real scipy Gamma fit, no mocking — the actual
    calibration/scoring pipeline end to end on a small synthetic corpus."""

    @staticmethod
    @pytest.fixture(scope="class")
    def calibrated_guardrail():
        guardrail = GammaGuardrail()
        guardrail.calibrate(SYNTHETIC_REFERENCE_CORPUS)
        return guardrail

    def test_calibration_produces_valid_gamma_params(self, calibrated_guardrail):
        assert calibrated_guardrail.is_calibrated
        shape, loc, scale = calibrated_guardrail._gamma_params
        assert shape > 0
        assert scale > 0

    def test_on_topic_sentence_scores_higher_than_nonsense(self, calibrated_guardrail):
        on_topic = "Neural networks use backpropagation to compute parameter gradients."
        nonsense = "Purple elephants juggle spaghetti while riding unicycles on Tuesdays."
        scores = calibrated_guardrail.survival_score([on_topic, nonsense])
        assert scores[0] > scores[1]

    def test_reference_corpus_member_scores_relatively_high(self, calibrated_guardrail):
        """A sentence drawn from the calibration set itself should sit
        well within the fitted distribution's typical range."""
        member_score = calibrated_guardrail.survival_score([SYNTHETIC_REFERENCE_CORPUS[0]])[0]
        nonsense_score = calibrated_guardrail.survival_score(
            ["Bicycles dream in the color of forgotten Wednesday soup."]
        )[0]
        assert member_score > nonsense_score
