"""Find highlights in a match video from crowd noise.

Goals and big chances make the crowd (and commentators) loud for several
seconds. We measure loudness per second with ffmpeg, smooth it, and pick the
peaks that stand well above the match's typical level.
"""
import json
import re
import subprocess
from pathlib import Path


def loudness_per_second(video):
    """Momentary loudness (LUFS) per second via ffmpeg's ebur128 filter."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "verbose", "-i", str(video), "-vn", "-af", "ebur128=framelog=verbose", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    per_second = {}
    for m in re.finditer(r"t:\s*([\d.]+)\s+TARGET:.*?M:\s*(-?[\d.]+|-inf)", proc.stderr):
        t, value = float(m.group(1)), m.group(2)
        per_second.setdefault(int(t), []).append(-70.0 if value == "-inf" else max(-70.0, float(value)))
    return [max(per_second.get(s, [-70.0])) for s in range(max(per_second, default=-1) + 1)]


def find_highlights(loudness, window=5, min_gap=30, threshold=2.0, top=10):
    """Seconds where smoothed loudness is `threshold` standard deviations above the median."""
    n = len(loudness)
    if n == 0:
        return []
    smooth = [sum(loudness[max(0, i - window):i + window + 1]) / len(loudness[max(0, i - window):i + window + 1]) for i in range(n)]
    ordered = sorted(smooth)
    median = ordered[n // 2]
    mad = sorted(abs(v - median) for v in smooth)[n // 2] or 1.0
    scores = [(v - median) / (1.4826 * mad) for v in smooth]
    peaks = []
    for i in sorted(range(n), key=lambda i: -scores[i]):
        if scores[i] < threshold or len(peaks) >= top:
            break
        if all(abs(i - p) >= min_gap for p, _ in peaks):
            peaks.append((i, scores[i]))
    # keep moments comparable to the loudest one; stray noise bumps fall far below
    best = max((z for _, z in peaks), default=0)
    return sorted((i, z) for i, z in peaks if z >= 0.2 * best)


def cut_clips(video, peaks, out_dir, before=20, after=10):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, (second, _) in enumerate(peaks, 1):
        start = max(0, second - before)
        path = out / f"highlight_{i:02d}_{second // 60:02d}m{second % 60:02d}s.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(video),
                        "-t", str(before + after), "-c", "copy", str(path)], check=True)
        clips.append(str(path))
    return clips


def highlights(video, clips_dir=None, top=10):
    peaks = find_highlights(loudness_per_second(video), top=top)
    result = [{"time": f"{s // 60:02d}:{s % 60:02d}", "second": s, "excitement": round(z, 1)} for s, z in peaks]
    if clips_dir:
        for r, clip in zip(result, cut_clips(video, peaks, clips_dir)):
            r["clip"] = clip
    return result


if __name__ == "__main__":
    import sys

    print(json.dumps(highlights(sys.argv[1]), indent=2))
