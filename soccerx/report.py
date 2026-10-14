"""Turn a match's event stream into a written summary."""
from collections import defaultdict

from .xg import is_shot, shot_xg


def clock(e):
    """'45+2' style match minute."""
    if e["period"] == 5:
        return "pens"
    base = {1: 45, 2: 90, 3: 105, 4: 120}.get(e["period"], 0)
    minute = e["minute"] + 1
    return f"{base}+{minute - base}'" if e["period"] <= 4 and minute > base else f"{minute}'"


def analyze(match, events, xg_model):
    home, away = match["home_team"]["home_team_name"], match["away_team"]["away_team_name"]
    teams = [home, away]
    other = {home: away, away: home}
    xg = defaultdict(float)
    shots = defaultdict(int)
    on_target = defaultdict(int)
    passes = defaultdict(int)
    goals, cards, subs, chances, shootout = [], [], [], [], []
    windows = defaultdict(lambda: defaultdict(float))  # 15-minute xG windows

    for e in events:
        team = e.get("team", {}).get("name")
        kind = e["type"]["name"]
        if kind == "Pass" and "outcome" not in e.get("pass", {}):
            passes[team] += 1
        if is_shot(e):
            if e["period"] == 5:
                shootout.append((team, e["player"]["name"], e["shot"]["outcome"]["name"] == "Goal"))
                continue
            value = shot_xg(xg_model, e)
            xg[team] += value
            shots[team] += 1
            outcome = e["shot"]["outcome"]["name"]
            on_target[team] += outcome in ("Goal", "Saved", "Saved To Post")
            windows[min(e["minute"] // 15, 7)][team] += value
            if outcome == "Goal":
                goals.append({"team": team, "player": e["player"]["name"], "time": clock(e), "xg": value,
                              "penalty": e["shot"]["type"]["name"] == "Penalty"})
            elif value >= 0.3:
                chances.append({"team": team, "player": e["player"]["name"], "time": clock(e), "xg": value, "outcome": outcome})
        if kind == "Own Goal Against":
            goals.append({"team": other[team], "player": e["player"]["name"] + " (own goal)", "time": clock(e), "xg": 0.0, "penalty": False})
        card = e.get("foul_committed", {}).get("card") or e.get("bad_behaviour", {}).get("card")
        if card:
            cards.append({"team": team, "player": e["player"]["name"], "time": clock(e), "card": card["name"]})
        if kind == "Substitution":
            subs.append({"team": team, "off": e["player"]["name"], "on": e["substitution"]["replacement"]["name"], "time": clock(e)})

    total_passes = sum(passes[t] for t in teams) or 1
    return {
        "home": home, "away": away,
        "score": (match["home_score"], match["away_score"]),
        "stage": match["competition_stage"]["name"], "date": match["match_date"],
        "competition": f"{match['competition']['competition_name']} {match['season']['season_name']}",
        "stadium": match.get("stadium", {}).get("name", ""),
        "xg": {t: round(xg[t], 2) for t in teams}, "shots": dict(shots), "on_target": dict(on_target),
        "pass_share": {t: round(100 * passes[t] / total_passes) for t in teams},
        "goals": goals, "cards": cards, "subs": subs, "shootout": shootout,
        "big_chances": sorted(chances, key=lambda c: -c["xg"])[:4],
        "windows": {w: dict(v) for w, v in sorted(windows.items())},
    }


def narrate(a):
    home, away = a["home"], a["away"]
    hs, as_ = a["score"]
    lines = [f"# {home} {hs}–{as_} {away}", f"{a['competition']}, {a['stage']} · {a['date']}" + (f" · {a['stadium']}" if a["stadium"] else ""), ""]

    if a["shootout"]:
        made = defaultdict(int)
        for team, _, scored in a["shootout"]:
            made[team] += scored
        winner = max(made, key=made.get)
        result = f"{winner} won {made[winner]}–{made[home if winner == away else away]} on penalties after a {hs}–{as_} draw."
    elif hs == as_:
        result = f"The match ended level at {hs}–{as_}."
    else:
        winner, loser = (home, away) if hs > as_ else (away, home)
        result = f"{winner} beat {loser} {max(hs, as_)}–{min(hs, as_)}."

    xh, xa = a["xg"][home], a["xg"][away]
    if abs(xh - xa) < 0.4:
        balance = "an evenly matched game by chance quality"
    else:
        better = home if xh > xa else away
        balance = f"{better} created the better chances"
    lines.append(f"{result} By expected goals it was {xh:.2f}–{xa:.2f}, {balance}; "
                 f"{home} had {a['pass_share'][home]}% of completed passes.")

    # who over/under-performed their chances
    for team, scored in ((home, hs), (away, as_)):
        diff = scored - a["xg"][team]
        if abs(diff) >= 1.0:
            lines.append(f"{team} {'outscored' if diff > 0 else 'fell short of'} their {a['xg'][team]:.2f} xG by {abs(diff):.1f} goals.")

    # momentum: which 15-minute spells were dominated
    spells = []
    for w, vals in a["windows"].items():
        h, aw = vals.get(home, 0), vals.get(away, 0)
        if max(h, aw) >= 0.5 and abs(h - aw) >= 0.4:
            spells.append(f"{w * 15}–{w * 15 + 15}' ({home if h > aw else away}, {max(h, aw):.1f} xG)")
    if spells:
        lines.append("Strongest spells: " + ", ".join(spells) + ".")

    lines += ["", "## Goals"]
    for g in a["goals"]:
        note = " (pen)" if g["penalty"] else (f", a {g['xg']:.2f} xG chance" if g["xg"] else "")
        lines.append(f"- {g['time']} {g['player']} ({g['team']}){note}")
    if a["shootout"]:
        lines.append("- Shootout: " + ", ".join(f"{p} {'✓' if s else '✗'}" for _, p, s in a["shootout"]))
    if a["big_chances"]:
        lines += ["", "## Big chances missed"]
        for c in a["big_chances"]:
            lines.append(f"- {c['time']} {c['player']} ({c['team']}), {c['xg']:.2f} xG, {c['outcome'].lower()}")
    if a["cards"]:
        lines += ["", "## Cards"] + [f"- {c['time']} {c['player']} ({c['team']}): {c['card']}" for c in a["cards"]]
    lines += ["", "## Stats", "| | " + home + " | " + away + " |", "| --- | --- | --- |"]
    for label, key in (("xG", "xg"), ("Shots", "shots"), ("On target", "on_target"), ("Pass share %", "pass_share")):
        lines.append(f"| {label} | {a[key].get(home, 0)} | {a[key].get(away, 0)} |")
    return "\n".join(lines) + "\n"
