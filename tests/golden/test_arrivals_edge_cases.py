from __future__ import annotations

import math

from iwc.analyze.arrivals import analyze_arrivals


def test_arrivals_single_request_no_nonsense():
    s = analyze_arrivals([0])
    assert s.n == 1
    assert s.duration_s == 0.0
    assert math.isnan(s.mean_rps)
    assert math.isnan(s.burstiness_cv)


def test_arrivals_identical_timestamps_no_div0():
    s = analyze_arrivals([0, 0])
    assert s.n == 2
    assert s.duration_s == 0.0
    assert math.isnan(s.mean_rps)
    assert math.isnan(s.burstiness_cv)


def test_arrivals_two_points_ok():
    s = analyze_arrivals([0, 100])
    assert s.n == 2
    assert s.duration_s == 0.1
    assert s.mean_rps > 0
