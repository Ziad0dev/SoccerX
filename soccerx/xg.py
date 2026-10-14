"""Expected goals: a small neural network trained on StatsBomb shots."""
import math
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import events_many, matches

MODEL = Path(__file__).resolve().parent.parent / "models" / "xg.pkl"
PENALTY_XG = 0.76  # long-run penalty conversion; too few penalties to learn
GOAL_X, POST_LOW, POST_HIGH = 120.0, 36.0, 44.0  # StatsBomb pitch is 120 x 80

BODY = ["Right Foot", "Left Foot", "Head"]
TECHNIQUE = ["Normal", "Half Volley", "Volley", "Lob", "Overhead Kick", "Diving Header", "Backheel"]
PATTERN = ["Regular Play", "From Corner", "From Free Kick", "From Throw In", "From Counter", "From Goal Kick", "From Keeper", "From Kick Off"]


def is_shot(e):
    return e["type"]["name"] == "Shot"


def is_penalty(e):
    return e["shot"]["type"]["name"] == "Penalty"


def features(e):
    x, y = e["location"][:2]
    dx, dy = GOAL_X - x, 40.0 - y
    distance = math.hypot(dx, dy)
    # angle the goal mouth subtends from the shot location
    a1 = math.atan2(POST_HIGH - y, dx)
    a2 = math.atan2(POST_LOW - y, dx)
    angle = abs(a1 - a2)
    s = e["shot"]
    one_hot = lambda value, options: [float(value == o) for o in options]  # noqa: E731
    return [
        distance, angle, math.log1p(distance), x, abs(dy),
        *freeze_frame_features(x, y, s.get("freeze_frame", [])),
        float(s.get("first_time", False)), float(e.get("under_pressure", False)),
        float(s.get("one_on_one", False)), float(s["type"]["name"] == "Free Kick"),
        *one_hot(s["body_part"]["name"], BODY),
        *one_hot(s["technique"]["name"], TECHNIQUE),
        *one_hot(e["play_pattern"]["name"], PATTERN),
    ]


def _in_cone(px, py, x, y):
    """Is (px, py) inside the triangle between the shooter and the two posts?"""
    def side(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    d1 = side(x, y, GOAL_X, POST_LOW, px, py)
    d2 = side(GOAL_X, POST_LOW, GOAL_X, POST_HIGH, px, py)
    d3 = side(GOAL_X, POST_HIGH, x, y, px, py)
    neg, pos = (d1 < 0) or (d2 < 0) or (d3 < 0), (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def freeze_frame_features(x, y, frame):
    """Defenders blocking the goal, keeper position, nearest opponent (StatsBomb freeze frames)."""
    opponents = [p for p in frame if not p["teammate"]]
    keeper = next((p for p in opponents if p["position"]["name"] == "Goalkeeper"), None)
    blockers = sum(_in_cone(*p["location"][:2], x, y) for p in opponents if p is not keeper)
    nearest = min((math.hypot(p["location"][0] - x, p["location"][1] - y) for p in opponents), default=10.0)
    if keeper:
        kx, ky = keeper["location"][:2]
        keeper_off_line = GOAL_X - kx
        keeper_in_cone = float(_in_cone(kx, ky, x, y))
        keeper_dist = math.hypot(kx - x, ky - y)
    else:  # keeper out of position / not in frame
        keeper_off_line, keeper_in_cone, keeper_dist = 20.0, 0.0, 30.0
    return [float(blockers), min(nearest, 10.0), keeper_off_line, keeper_in_cone, keeper_dist, float(bool(frame))]


def shots_from(event_lists):
    X, y, sb = [], [], []
    for evs in event_lists:
        for e in evs:
            if is_shot(e) and not is_penalty(e) and e.get("location"):
                X.append(features(e))
                y.append(int(e["shot"]["outcome"]["name"] == "Goal"))
                sb.append(e["shot"].get("statsbomb_xg", np.nan))
    return np.array(X), np.array(y), np.array(sb)


def train(train_comps=("wc2018", "euro2020", "euro2024"), test_comp="wc2022"):
    def load(comps):
        ids = [m["match_id"] for c in comps for m in matches(c)]
        print(f"loading events for {len(ids)} matches ({', '.join(comps)})...")
        return list(events_many(ids).values())

    X, y, _ = shots_from(load(train_comps))
    Xt, yt, sbt = shots_from(load([test_comp]))
    model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2,
                                                          max_iter=600, early_stopping=True, random_state=0))
    model.fit(X, y)
    p = model.predict_proba(Xt)[:, 1]
    print(f"trained on {len(y)} shots ({y.mean():.1%} goals), tested on {len(yt)} {test_comp} shots")
    print(f"  ours:      AUC {roc_auc_score(yt, p):.3f}  log loss {log_loss(yt, p):.3f}  Brier {brier_score_loss(yt, p):.4f}")
    print(f"  StatsBomb: AUC {roc_auc_score(yt, sbt):.3f}  log loss {log_loss(yt, sbt):.3f}  Brier {brier_score_loss(yt, sbt):.4f}")
    print(f"  total xG {p.sum():.1f} vs actual goals {yt.sum()}")
    MODEL.parent.mkdir(exist_ok=True)
    MODEL.write_bytes(pickle.dumps(model))
    print(f"saved {MODEL}")
    return model


def load_model():
    if not MODEL.exists():
        raise SystemExit("train the xG model first: python -m soccerx train-xg")
    return pickle.loads(MODEL.read_bytes())


def shot_xg(model, e):
    if is_penalty(e):
        return PENALTY_XG
    return float(model.predict_proba([features(e)])[0, 1])
