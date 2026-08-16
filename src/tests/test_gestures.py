"""Arm-level gesture recognition over synthetic keypoint windows.

Run from src/:  python -m pytest tests -q
"""
from app.gestures import detect_gestures


def _kps(l_wrist=(85, 130), r_wrist=(115, 130),
         l_elbow=(85, 120), r_elbow=(115, 120)):
    """Upright skeleton; shoulders fixed at y=100."""
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[5] = [90.0, 100.0, 0.9]      # left shoulder
    kps[6] = [110.0, 100.0, 0.9]     # right shoulder
    kps[7] = [float(l_elbow[0]), float(l_elbow[1]), 0.9]
    kps[8] = [float(r_elbow[0]), float(r_elbow[1]), 0.9]
    kps[9] = [float(l_wrist[0]), float(l_wrist[1]), 0.9]
    kps[10] = [float(r_wrist[0]), float(r_wrist[1]), 0.9]
    return kps


def test_empty_and_all_none():
    assert detect_gestures([]) == []
    assert detect_gestures([None, None, None]) == []


def test_hands_down_is_nothing():
    assert detect_gestures([_kps() for _ in range(6)]) == []


def test_sustained_raise_fires_single_frame_does_not():
    up = _kps(l_wrist=(85, 80))
    down = _kps()
    assert detect_gestures([down, up, down, down, down, down]) == []
    assert detect_gestures([down, up, up, up, down, down]) == ["hand_raised"]


def test_both_hands_up():
    up = _kps(l_wrist=(85, 80), r_wrist=(115, 80))
    out = detect_gestures([up, up, up, _kps()])
    assert out == ["both_hands_up"]          # implies (and hides) hand_raised


def test_wave_swings_across_elbow():
    # Raised wrist alternating 20px left/right of the elbow (x=85).
    frames = [_kps(l_wrist=(85 + s, 80))
              for s in (20, -20, 20, -20, 20, -20)]
    out = detect_gestures(frames)
    assert "wave" in out
    assert "hand_raised" not in out          # implied by the wave


def test_deadband_swallows_jitter_wave():
    # 1px oscillation is keypoint noise, not a wave - reads as raised only.
    frames = [_kps(l_wrist=(85 + s, 80))
              for s in (1, -1, 1, -1, 1, -1)]
    assert detect_gestures(frames) == ["hand_raised"]


def test_low_conf_wrist_is_ignored():
    up = _kps(l_wrist=(85, 80))
    up[9][2] = 0.1                            # wrist below KP_MIN_CONF
    assert detect_gestures([up] * 6) == []
