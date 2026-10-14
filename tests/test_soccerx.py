from soccerx.report import clock
from soccerx.video import find_highlights
from soccerx.xg import _in_cone, freeze_frame_features


def test_clock():
    assert clock({"period": 1, "minute": 46}) == "45+2'"
    assert clock({"period": 2, "minute": 60}) == "61'"
    assert clock({"period": 5, "minute": 121}) == "pens"


def test_cone_and_freeze_frame():
    assert _in_cone(110, 40, 100, 40)       # between shooter and goal centre
    assert not _in_cone(110, 10, 100, 40)   # off to the side
    frame = [
        {"teammate": False, "location": [118, 40], "position": {"name": "Goalkeeper"}},
        {"teammate": False, "location": [105, 40], "position": {"name": "Center Back"}},
        {"teammate": True, "location": [104, 30], "position": {"name": "Striker"}},
    ]
    blockers, nearest, off_line, keeper_in_cone, _, has_frame = freeze_frame_features(100, 40, frame)
    assert blockers == 1 and nearest == 5 and off_line == 2 and keeper_in_cone == 1 and has_frame == 1


def test_highlights_find_loud_moments():
    loudness = [-30.0 + (i % 3) * 0.3 for i in range(300)]
    for s in list(range(100, 108)) + list(range(220, 225)):
        loudness[s] = -12.0
    peaks = [s for s, _ in find_highlights(loudness)]
    assert len(peaks) == 2 and 98 <= peaks[0] <= 110 and 218 <= peaks[1] <= 227
