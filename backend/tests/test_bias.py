"""The decision bias: the term that can move a two-option boundary when temperature cannot.

Calibrating v0.7 showed the gap this fills. Its yes/no discrimination was the best we had measured
(AUC 0.622, best-threshold accuracy 0.720) but it scored 0.627 at the conventional 0.5 cut, because
its boundary sat at 0.621. Temperature cannot correct that, for the reason the first test pins down.
"""

from __future__ import annotations

import math

from systemone_builder.kenning.calibrate import scores_from
from systemone_builder.kenning.train import _softmax, fit_biases, fit_temperatures, metrics


def test_temperature_cannot_move_a_two_option_decision():
    """The premise: for noul, p >= 0.5 is the same test as score_yes >= score_no, whatever T is."""
    scores = [0.8, 1.1]  # "no" wins
    for t in (0.05, 0.5, 1.0, 2.0, 20.0):
        p_yes = _softmax(scores, t)[0]
        assert p_yes < 0.5, f"T={t} changed the decision, which should be impossible"
    # and a losing option never wins back at any temperature
    for t in (0.05, 1.0, 20.0):
        assert (_softmax([1.1, 0.8], t)[0] >= 0.5) is True


def test_bias_can_move_the_decision():
    scores = [0.8, 1.1]
    assert _softmax(scores, 1.0)[0] < 0.5
    assert _softmax(scores, 1.0, bias=0.5)[0] > 0.5          # enough bias flips it
    assert _softmax(scores, 1.0, bias=-0.5)[0] < 0.5         # the other way keeps it


def test_zero_bias_is_exactly_the_old_behaviour():
    for scores in ([0.3, 0.7], [2.0, -1.0], [0.0, 0.0], [1.0, 2.0, 3.0]):
        for t in (0.5, 1.0, 1.8221):
            assert _softmax(scores, t, 0.0) == _softmax(scores, t)


def test_softmax_still_normalises_with_a_bias():
    p = _softmax([0.4, 1.3], 1.4, bias=-0.9)
    assert math.isclose(sum(p), 1.0, rel_tol=1e-12)
    assert all(0.0 <= x <= 1.0 for x in p)


def _noul_rows(true_bias: float, n: int = 400) -> list[tuple[str, list[float], list[float]]]:
    """Synthetic noul rows whose labels are generated with a known decision offset."""
    rows = []
    for i in range(n):
        gap = -1.0 + 2.0 * (i / (n - 1))                 # score_yes - score_no, spread over [-1, 1]
        yes = 1 if gap + true_bias >= 0 else 0
        rows.append(("noul", [gap, 0.0], [float(yes), float(1 - yes)]))
    return rows


def test_fit_biases_recovers_the_injected_decision_boundary():
    """The fitted value is in post-temperature units, so check the boundary it implies, not the number.

    Labels are yes when ``gap >= -true_offset``, and the served decision flips when
    ``gap / T + bias == 0``, i.e. at ``gap == -bias * T``. Asserting on the raw bias instead would
    just be asserting the fitted temperature, which is not what this term is for.
    """
    for true_offset in (-0.4, 0.0, 0.5):
        raw = _noul_rows(true_offset)
        temps = fit_temperatures(raw)
        bias = fit_biases(raw, temps)["noul"]
        boundary = -bias * temps["noul"]
        assert abs(boundary - (-true_offset)) < 0.15, (
            f"offset {true_offset}: boundary landed at {boundary:.3f}, wanted {-true_offset:.3f} "
            f"(T={temps['noul']}, bias={bias})")


def test_fitting_a_bias_improves_accuracy_where_temperature_cannot():
    raw = _noul_rows(0.5)
    temps = fit_temperatures(raw)
    before = metrics(raw, temps)["noul"]["accuracy"]
    biases = fit_biases(raw, temps)
    after = metrics(raw, temps, biases)["noul"]["accuracy"]
    assert after > before, f"bias did not help: {before} -> {after}"


def test_fit_biases_only_touches_two_option_questions():
    raw = _noul_rows(0.3) + [("choice", [0.2, 0.9, 0.1], [0.0, 1.0, 0.0])] * 50 \
        + [("score", [0.5, 0.4, 0.1], [1.0, 0.0, 0.0])] * 50
    got = fit_biases(raw, fit_temperatures(raw))
    assert set(got) == {"noul"}, f"bias should be noul-only, got {sorted(got)}"


def test_scores_from_removes_a_bias_the_server_already_applied():
    """Otherwise a refit would sit on top of the served bias and double it."""
    scores, t, b = [0.7, 1.2], 1.5, 0.6
    served = _softmax(scores, t, b)[0]
    answer = {"type": "noul", "noul": served}
    _, recovered = scores_from(answer, t, b)
    # recovery is up to a shared additive constant, so compare the gap that actually decides
    assert math.isclose(recovered[0] - recovered[1], scores[0] - scores[1], abs_tol=1e-9)


def test_scores_from_without_a_bias_is_unchanged():
    scores, t = [0.7, 1.2], 1.5
    answer = {"type": "noul", "noul": _softmax(scores, t)[0]}
    _, recovered = scores_from(answer, t)
    assert math.isclose(recovered[0] - recovered[1], scores[0] - scores[1], abs_tol=1e-9)
