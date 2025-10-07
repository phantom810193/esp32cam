import pytest

from backend.prediction import _blend_with_prior, _softmax


def _is_prob_vec(probs):
    return abs(sum(probs) - 1) < 1e-8 and all(0 <= value <= 1 for value in probs)


def test_softmax_basic():
    scores = [1.2, 0.8, 0.3, -0.5]
    probs = _softmax(scores, tau=1.6)
    assert _is_prob_vec(probs)


def test_softmax_temperature():
    scores = [3.0, 1.0, 0.0]
    probs_low = _softmax(scores, tau=0.6)
    probs_high = _softmax(scores, tau=3.0)
    assert max(probs_low) > max(probs_high)


def test_blend_with_prior_behaviour():
    p_soft = [0.9, 0.1]
    prior = [0.5, 0.5]
    small_n = _blend_with_prior(p_soft, prior, n=0, k=10)
    large_n = _blend_with_prior(p_soft, prior, n=1000, k=10)
    assert small_n[0] < large_n[0]
