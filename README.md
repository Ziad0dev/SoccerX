# SoccerX
Sports match video to text summarization using neural network

Two halves of the problem:

**1. Match events → written report** (`python -m soccerx report "wc2022 final"`)
Uses [StatsBomb open data](https://github.com/statsbomb/open-data) event streams (World Cups 2018/2022, Euros 2020/2024) and generates a report: result (incl. shootouts), expected-goals balance, over/under-performance, the strongest 15-minute spells, goals with chance quality, big chances missed, cards and a stats table.

Expected goals comes from our own **neural network** (`soccerx/xg.py`, an MLP) trained on 4,176 shots from WC 2018 / Euro 2020 / Euro 2024 and tested on all 1,430 shots of WC 2022. Features: distance and angle to goal, body part, technique, play pattern, and **freeze-frame** features (defenders in the shooting cone, keeper position, nearest opponent).

| WC 2022 test shots | ROC AUC | log loss | Brier |
| --- | --- | --- | --- |
| SoccerX MLP | 0.777 | 0.294 | 0.085 |
| StatsBomb's commercial xG | 0.815 | 0.267 | 0.076 |

**2. Match video → highlights** (`python -m soccerx highlights match.mp4 --clips out/`)
Measures loudness per second with ffmpeg (EBU R128) and finds crowd/commentator roars that stand far above the match's typical level, then cuts clips around them (20 s before, 10 s after). Tested on synthetic audio with known roar times.

```sh
nix-shell                                   # numpy, scikit-learn, ffmpeg
python -m soccerx train-xg
python -m soccerx list wc2022
python -m soccerx report "euro2024 spain england" -o report.md
python -m soccerx highlights match.mp4 --clips highlights/
python -m pytest tests
```

Next step toward true video-to-text: align video highlights to event timestamps (kick-off offset) and caption frames with a vision-language model.
