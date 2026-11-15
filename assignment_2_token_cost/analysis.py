"""Pairwise comparison, summary statistics, quality-check sheet and charts."""
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results.csv"
META = HERE / "run_meta.json"
COMPARISON = HERE / "comparison.csv"


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def bootstrap_ci(values, iterations=5000, seed=42):
    """95% percentile bootstrap interval of the mean."""
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values)))
             for _ in range(iterations)]
    return percentile(means, 0.025), percentile(means, 0.975)


def _ok(row):
    return str(row.get("input_count_ok", "True")) == "True" and int(row["input_tokens"]) > 0


def build_comparison(rows, baseline=0):
    """One dict per pair. Pairs with a missing/zero input count are marked invalid."""
    by_pair = defaultdict(dict)
    for row in rows:
        by_pair[row["id"]][row["language"]] = row
    out = []
    for pid, pair in sorted(by_pair.items()):
        if "english" not in pair or "bangla" not in pair:
            continue
        en, bn = pair["english"], pair["bangla"]
        en_in, bn_in = int(en["input_tokens"]), int(bn["input_tokens"])
        en_cost, bn_cost = float(en["cost_usd"]), float(bn["cost_usd"])
        valid = _ok(en) and _ok(bn)
        ratio = bn_in / en_in if valid else None
        en_c, bn_c = en_in - baseline, bn_in - baseline
        content = bn_c / en_c if valid and en_c > 0 and bn_c > 0 else None
        out.append({
            "id": pid, "category": en["category"], "valid": valid,
            "english_input_tokens": en_in, "bangla_input_tokens": bn_in,
            "input_token_ratio_bn_over_en": ratio,
            "content_only_ratio_bn_over_en": content,
            "english_output_tokens": int(en["output_tokens"]),
            "bangla_output_tokens": int(bn["output_tokens"]),
            "english_cost_usd": en_cost, "bangla_cost_usd": bn_cost,
            "total_cost_ratio_bn_over_en": bn_cost / en_cost if en_cost else None,
            "english_latency_s": float(en["latency_s"]),
            "bangla_latency_s": float(bn["latency_s"]),
            "english_truncated": str(en.get("truncated")) == "True",
            "bangla_truncated": str(bn.get("truncated")) == "True",
        })
    return out


def spread(values):
    if not values:
        return {}
    lo, hi = bootstrap_ci(values)
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": min(values),
            "max": max(values), "ci95_low": lo, "ci95_high": hi}


def summarize(rows, comparison, baseline=0):
    valid = [c for c in comparison if c["valid"]]
    en = [r for r in rows if r["language"] == "english"]
    bn = [r for r in rows if r["language"] == "bangla"]
    tot = lambda rs, k: sum(int(r[k]) for r in rs)
    cost = lambda rs: sum(float(r["cost_usd"]) for r in rs)
    cats = defaultdict(list)
    for c in valid:
        cats[c["category"]].append(c["input_token_ratio_bn_over_en"])
    return {
        "baseline_input_tokens": baseline,
        "pairs": len(comparison), "valid_pairs": len(valid),
        "invalid_pairs": [c["id"] for c in comparison if not c["valid"]],
        "input_tokens": {"english": tot(en, "input_tokens"), "bangla": tot(bn, "input_tokens")},
        "output_tokens": {"english": tot(en, "output_tokens"), "bangla": tot(bn, "output_tokens")},
        "cost_usd": {"english": cost(en), "bangla": cost(bn),
                     "ratio_bn_over_en": cost(bn) / cost(en) if cost(en) else None},
        "input_ratio": spread([c["input_token_ratio_bn_over_en"] for c in valid]),
        "content_only_ratio": spread([c["content_only_ratio_bn_over_en"] for c in valid
                                      if c["content_only_ratio_bn_over_en"]]),
        "pooled_input_ratio": (sum(c["bangla_input_tokens"] for c in valid)
                               / sum(c["english_input_tokens"] for c in valid)) if valid else None,
        "latency_s": {
            "english_mean": statistics.mean(float(r["latency_s"]) for r in en),
            "english_median": statistics.median(float(r["latency_s"]) for r in en),
            "bangla_mean": statistics.mean(float(r["latency_s"]) for r in bn),
            "bangla_median": statistics.median(float(r["latency_s"]) for r in bn)},
        "truncated_by_num_predict": {
            "english": sum(str(r.get("truncated")) == "True" for r in en),
            "bangla": sum(str(r.get("truncated")) == "True" for r in bn)},
        "by_category": {k: {"n": len(v), "mean_ratio": statistics.mean(v),
                            "median_ratio": statistics.median(v)}
                        for k, v in sorted(cats.items(), key=lambda kv: -statistics.mean(kv[1]))},
        "min_ratio_pair": min(valid, key=lambda c: c["input_token_ratio_bn_over_en"])["id"] if valid else None,
        "max_ratio_pair": max(valid, key=lambda c: c["input_token_ratio_bn_over_en"])["id"] if valid else None,
    }


def write_comparison(comparison, path=COMPARISON):
    def fmt(v):
        return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else v)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        for c in comparison:
            writer.writerow({k: fmt(v) for k, v in c.items()})


def write_quality_sheet(rows, path, n=10):
    """10 pairs spread over categories, for a HUMAN to judge Bangla vs English."""
    by_pair = defaultdict(dict)
    for r in rows:
        by_pair[r["id"]][r["language"]] = r
    ids = sorted(by_pair)
    step = max(1, len(ids) // n)
    chosen = ids[::step][:n]
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "english_prompt", "english_output",
                    "bangla_prompt", "bangla_output",
                    "bangla_as_correct_and_useful (yes/no/partly)", "your_note"])
        for pid in chosen:
            en, bn = by_pair[pid]["english"], by_pair[pid]["bangla"]
            w.writerow([pid, en["category"], en["prompt"], en["output"],
                        bn["prompt"], bn["output"], "", ""])


def _fmt_stats(s):
    return (f"mean {s['mean']:.2f}, median {s['median']:.2f}, min {s['min']:.2f}, "
            f"max {s['max']:.2f}, 95% CI [{s['ci95_low']:.2f}, {s['ci95_high']:.2f}]")


def render_markdown(s):
    lines = ["# Assignment 2 summary", "",
             f"Template + system-prompt baseline: **{s['baseline_input_tokens']}** input tokens "
             "(measured with a 1-character prompt).",
             f"Valid pairs: {s['valid_pairs']}/{s['pairs']}"
             + (f" (excluded, missing input count: {s['invalid_pairs']})" if s['invalid_pairs'] else ""),
             "", "| Measure | English | Bangla |", "|---|---|---|",
             f"| Input tokens (total) | {s['input_tokens']['english']} | {s['input_tokens']['bangla']} |",
             f"| Output tokens (total) | {s['output_tokens']['english']} | {s['output_tokens']['bangla']} |",
             f"| Cost USD (notional) | {s['cost_usd']['english']:.6f} | {s['cost_usd']['bangla']:.6f} |",
             f"| Latency mean / median (s) | {s['latency_s']['english_mean']:.2f} / {s['latency_s']['english_median']:.2f} "
             f"| {s['latency_s']['bangla_mean']:.2f} / {s['latency_s']['bangla_median']:.2f} |",
             f"| Answers cut by max tokens | {s['truncated_by_num_predict']['english']} | {s['truncated_by_num_predict']['bangla']} |",
             "", f"- Total cost ratio (Bangla / English): **{s['cost_usd']['ratio_bn_over_en']:.3f}**",
             f"- Pooled input ratio (sum of tokens): {s['pooled_input_ratio']:.3f}" if s['pooled_input_ratio'] else "",
             f"- Per-pair input ratio: {_fmt_stats(s['input_ratio'])}"]
    if s["content_only_ratio"]:
        lines.append(f"- Per-pair ratio after removing the fixed baseline: {_fmt_stats(s['content_only_ratio'])}")
    lines += ["", "## By category (per-pair input ratio, largest gap first)",
              "| category | pairs | mean | median |", "|---|---|---|---|"]
    for k, v in s["by_category"].items():
        lines.append(f"| {k} | {v['n']} | {v['mean_ratio']:.2f} | {v['median_ratio']:.2f} |")
    lines += ["", f"Lowest-ratio pair: {s['min_ratio_pair']}; highest-ratio pair: {s['max_ratio_pair']} "
              "(look at these for the 'surprising prompt' in your report)."]
    return "\n".join(lines) + "\n"


def make_charts(comparison, summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, accent = "#1f2937", "#6b7280", "#2563eb"
    valid = [c for c in comparison if c["valid"]]
    ratios = [c["input_token_ratio_bn_over_en"] for c in valid]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(range(len(ratios)), ratios, color=accent, width=0.7)
    ax.axhline(1, color=muted, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([c["id"] for c in valid], rotation=90, fontsize=7)
    ax.set_ylabel("Bangla / English input tokens", color=ink)
    ax.set_title("Input token ratio per aligned prompt pair (1.0 = same cost)", color=ink, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "input_token_ratio.png", dpi=160)
    plt.close(fig)

    cats = list(summary["by_category"])
    means = [summary["by_category"][c]["mean_ratio"] for c in cats]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(cats[::-1], means[::-1], color=accent, height=0.6)
    ax.axvline(1, color=muted, linestyle="--", linewidth=1)
    for i, v in enumerate(means[::-1]):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8, color=ink)
    ax.set_xlabel("Mean Bangla / English input-token ratio", color=ink)
    ax.set_title("Input token ratio by category", color=ink, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "input_token_ratio_by_category.png", dpi=160)
    plt.close(fig)


def main(results=RESULTS, meta=META, out_dir=HERE):
    out_dir = Path(out_dir)
    results = Path(results)
    if not results.exists():
        raise SystemExit("No results found. Run the runner first: "
                         "python -m assignment_2_token_cost.runner")
    with results.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("results.csv is empty. Run the runner first.")
    baseline = 0
    if Path(meta).exists():
        baseline = json.loads(Path(meta).read_text(encoding="utf-8")).get("baseline_input_tokens", 0)
    comparison = build_comparison(rows, baseline)
    if not comparison:
        raise SystemExit("No complete English/Bangla pairs in results.csv.")
    write_comparison(comparison, out_dir / "comparison.csv")
    summary = summarize(rows, comparison, baseline)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    text = render_markdown(summary)
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    write_quality_sheet(rows, out_dir / "quality_check.csv")
    make_charts(comparison, summary, out_dir)
    print(text)
    print("Wrote comparison.csv, summary.md/json, quality_check.csv and 2 charts to", out_dir)


if __name__ == "__main__":
    main()
