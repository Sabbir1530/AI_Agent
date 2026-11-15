"""Summarise run_log.jsonl: pass@1, pass^3, failure types, tokens, time.

Usage: python -m assignment_1_sql_agent.report [run_log.jsonl] [--before OLD.jsonl]
pass@1 = share of all runs that are correct (average single-try success).
pass^k = share of questions answered correctly in ALL k trials.
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from assignment_1_sql_agent.grading import classify_failure, is_correct

HERE = Path(__file__).parent


def load(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def metrics(records):
    by_q = defaultdict(list)
    for r in records:
        by_q[r["id"]].append(is_correct(r))
    k = min(len(v) for v in by_q.values())
    return {
        "questions": len(by_q), "runs": len(records), "k": k,
        "pass@1": sum(map(sum, by_q.values())) / len(records),
        "pass^k": sum(all(v) for v in by_q.values()) / len(by_q),
        "by_question": {q: (sum(v), len(v)) for q, v in by_q.items()},
    }


def render(records, before=None):
    m = metrics(records)
    lines = [f"# Assignment 1 results ({m['questions']} questions x {m['k']} trials)", "",
             f"- pass@1 (all runs): **{m['pass@1']:.3f}**",
             f"- pass^{m['k']} (all trials correct): **{m['pass^k']:.3f}**"]
    if before:
        b = metrics(before)
        lines += ["", "## Before vs after", "| | before | after |", "|---|---|---|",
                  f"| pass@1 | {b['pass@1']:.3f} | {m['pass@1']:.3f} |",
                  f"| pass^{m['k']} | {b['pass^k']:.3f} | {m['pass^k']:.3f} |"]
    lines += ["", "## Per question (correct trials / trials)", "| id | kind | correct |", "|---|---|---|"]
    kind = {r["id"]: r.get("kind", "") for r in records}
    for q, (c, n) in m["by_question"].items():
        lines.append(f"| {q} | {kind[q]} | {c}/{n} |")
    fails = [r for r in records if not is_correct(r)]
    types = Counter(classify_failure(r) for r in fails)
    lines += ["", "## Failure types (most common first)"]
    if not fails:
        lines.append("No failures.")
    for ftype, n in types.most_common():
        ex = next(r for r in fails if classify_failure(r) == ftype)
        lines += [f"- **{ftype}**: {n} run(s). Example {ex['id']} trial {ex['trial']}: "
                  f"expected `{ex['expected']}`, got `{ex['final_answer'][:120]}` "
                  f"(status {ex['status']}, tools {ex['tool_calls']})"]
    toks = [r["tokens"] for r in records]
    times = [r["time_s"] for r in records]
    lines += ["", "## Cost of a run",
              f"- tokens per run: mean {statistics.mean(toks):.0f}, max {max(toks)}",
              f"- seconds per run: mean {statistics.mean(times):.1f}, max {max(times):.1f}",
              f"- statuses: {dict(Counter(r['status'] for r in records))}"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=str(HERE / "run_log.jsonl"))
    ap.add_argument("--before", help="earlier run log, to show before/after numbers")
    args = ap.parse_args(argv)
    text = render(load(args.log), load(args.before) if args.before else None)
    (HERE / "results_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
