"""
    python -m soccerx train-xg                       # train the xG network, compare with StatsBomb's model
    python -m soccerx report "wc2022 final"          # written match report
    python -m soccerx report "euro2024 Spain England" -o report.md
    python -m soccerx highlights match.mp4 --clips out/
    python -m soccerx list wc2022
"""
import argparse
import json

from . import data, report, video, xg

parser = argparse.ArgumentParser(prog="soccerx", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = parser.add_subparsers(dest="cmd", required=True)
sub.add_parser("train-xg")
p = sub.add_parser("report")
p.add_argument("match", help="'wc2022 final', 'euro2024 spain england' or a match id")
p.add_argument("-o", "--out")
p = sub.add_parser("highlights")
p.add_argument("video")
p.add_argument("--clips")
p.add_argument("--top", type=int, default=10)
p = sub.add_parser("list")
p.add_argument("competition", choices=list(data.COMPETITIONS))
args = parser.parse_args()

if args.cmd == "train-xg":
    xg.train()
elif args.cmd == "report":
    match = data.find_match(args.match)
    text = report.narrate(report.analyze(match, data.events(match["match_id"]), xg.load_model()))
    if args.out:
        open(args.out, "w").write(text)
    print(text)
elif args.cmd == "highlights":
    print(json.dumps(video.highlights(args.video, args.clips, args.top), indent=2))
elif args.cmd == "list":
    for m in sorted(data.matches(args.competition), key=lambda m: m["match_date"]):
        print(f"{m['match_id']}  {m['match_date']}  {m['competition_stage']['name']:16s} "
              f"{m['home_team']['home_team_name']} {m['home_score']}-{m['away_score']} {m['away_team']['away_team_name']}")
