"""StatsBomb open data (https://github.com/statsbomb/open-data), cached locally."""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE = Path(__file__).resolve().parent.parent / "data"

# (competition_id, season_id)
COMPETITIONS = {
    "wc2022": (43, 106),
    "wc2018": (43, 3),
    "euro2024": (55, 282),
    "euro2020": (55, 43),
}


def fetch(path):
    local = CACHE / path
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{BASE}/{path}", timeout=120) as resp:
            local.write_bytes(resp.read())
    return json.loads(local.read_text(encoding="utf-8"))


def matches(competition):
    comp, season = COMPETITIONS[competition]
    return fetch(f"matches/{comp}/{season}.json")


def events(match_id):
    return fetch(f"events/{match_id}.json")


def events_many(match_ids, workers=8):
    with ThreadPoolExecutor(workers) as pool:
        return dict(zip(match_ids, pool.map(events, match_ids)))


def find_match(query):
    """'wc2022 final', 'euro2024 Spain England', or a numeric match id."""
    if str(query).isdigit():
        for comp in COMPETITIONS:
            for m in matches(comp):
                if m["match_id"] == int(query):
                    return m
        raise SystemExit(f"match {query} not found")
    words = query.lower().split()
    comp = next((w for w in words if w in COMPETITIONS), "wc2022")
    rest = [w for w in words if w != comp]
    candidates = []
    for m in matches(comp):
        teams = f"{m['home_team']['home_team_name']} {m['away_team']['away_team_name']}".lower()
        stage = m["competition_stage"]["name"].lower()
        text = f"{teams} {stage} {m['match_date']}"
        if all(w in text for w in rest):
            # "final" should pick the Final, not a quarter-final
            exact_stage = " ".join(w for w in rest if w not in teams) == stage
            candidates.append((not exact_stage, m["match_date"], m))
    if not candidates:
        raise SystemExit(f"no {comp} match matches '{query}'")
    return min(candidates, key=lambda c: c[:2])[2]
