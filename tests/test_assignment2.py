"""Dataset, runner and analysis tests using a fake Ollama backend."""
import csv
import json
from assignment_2_token_cost import analysis, runner
from assignment_2_token_cost.runner import DATASET, load_dataset


def test_dataset_has_50_aligned_pairs_with_required_columns():
    rows = load_dataset(DATASET)
    assert len(rows) == 50
    assert list(rows[0]) == ["id", "category", "english", "bangla"]
    assert len({r["id"] for r in rows}) == 50
    assert all(r["english"].strip() and r["bangla"].strip() for r in rows)
    # Bangla block U+0980-U+09FF must appear in every Bangla prompt
    assert all(any("ঀ" <= ch <= "৿" for ch in r["bangla"]) for r in rows)
    assert len({r["category"] for r in rows}) >= 6
    lengths = [len(r["english"]) for r in rows]
    assert max(lengths) > 4 * min(lengths)  # lengths are varied


class FakeBackend:
    """Bangla prompts cost 3x the characters; records the settings it receives."""
    model = "fake:1b"

    def __init__(self, cached_ids=()):
        self.calls, self.cached = [], set(cached_ids)

    def generate(self, prompt, system="", **options):
        self.calls.append((prompt, system, options))
        bangla = any("ঀ" <= c <= "৿" for c in prompt)
        n_in = 10 + len(prompt) * (3 if bangla else 1) // 4
        resp = {"message": {"content": "answer"}, "prompt_eval_count": n_in,
                "eval_count": 8, "done_reason": "length" if len(prompt) > 200 else "stop"}
        if prompt in self.cached:
            del resp["prompt_eval_count"]  # simulate Ollama omitting cached count
        return resp


def small_dataset(tmp_path):
    p = tmp_path / "d.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "english", "bangla"])
        w.writerow(["P01", "question", "What is the capital of Bangladesh?", "বাংলাদেশের রাজধানী কী?"])
        w.writerow(["P02", "arithmetic", "What is 2 plus 2?", "2 এর সাথে 2 যোগ করলে কত হয়?"])
    return p


def test_runner_same_settings_for_both_languages_and_records_everything(tmp_path):
    backend = FakeBackend()
    res, meta = tmp_path / "r.csv", tmp_path / "m.json"
    runner.run(client=backend, dataset=small_dataset(tmp_path), results=res, meta=meta, verbose=False)
    rows = list(csv.DictReader(res.open(encoding="utf-8")))
    assert [(r["id"], r["language"]) for r in rows] == [
        ("P01", "english"), ("P01", "bangla"), ("P02", "english"), ("P02", "bangla")]
    assert list(rows[0]) == runner.FIELDS
    measured = [o for _, _, o in backend.calls[1:]]  # calls[0] is the baseline probe
    assert len(measured) == 4 and all(o == runner.SETTINGS for o in measured)
    assert {s for _, s, _ in backend.calls} == {runner.SYSTEM_PROMPT}
    assert all(r["output"] == "answer" and float(r["cost_usd"]) > 0 for r in rows)
    assert json.loads(meta.read_text())["baseline_input_tokens"] > 0


def test_runner_resume_skips_done_rows(tmp_path):
    ds, res, meta = small_dataset(tmp_path), tmp_path / "r.csv", tmp_path / "m.json"
    runner.run(client=FakeBackend(), dataset=ds, results=res, meta=meta, verbose=False)
    b2 = FakeBackend()
    runner.run(client=b2, dataset=ds, results=res, meta=meta, resume=True, verbose=False)
    assert len(b2.calls) == 1  # only the baseline probe
    assert len(list(csv.DictReader(res.open(encoding="utf-8")))) == 4


def test_cost_arithmetic_in_row():
    item = {"id": "P01", "category": "q", "english": "a", "bangla": "b"}
    row = runner.make_row(item, "english", {"message": {"content": "x"},
                          "prompt_eval_count": 1000, "eval_count": 500}, 1.5)
    assert float(row["cost_usd"]) == 0.00045 and row["latency_s"] == "1.5000"


def run_and_load(tmp_path, backend):
    res, meta = tmp_path / "r.csv", tmp_path / "m.json"
    runner.run(client=backend, dataset=small_dataset(tmp_path), results=res, meta=meta, verbose=False)
    return list(csv.DictReader(res.open(encoding="utf-8"))), json.loads(meta.read_text())["baseline_input_tokens"]


def test_comparison_ratios_and_summary(tmp_path):
    rows, base = run_and_load(tmp_path, FakeBackend())
    comp = analysis.build_comparison(rows, base)
    assert len(comp) == 2 and all(c["valid"] for c in comp)
    for c in comp:
        assert abs(c["input_token_ratio_bn_over_en"] - c["bangla_input_tokens"] / c["english_input_tokens"]) < 1e-9
        assert c["content_only_ratio_bn_over_en"] > c["input_token_ratio_bn_over_en"]  # baseline dilutes
    s = analysis.summarize(rows, comp, base)
    assert s["input_tokens"]["english"] == sum(c["english_input_tokens"] for c in comp)
    assert s["input_ratio"]["min"] <= s["input_ratio"]["median"] <= s["input_ratio"]["max"]
    assert s["input_ratio"]["ci95_low"] <= s["input_ratio"]["mean"] <= s["input_ratio"]["ci95_high"]


def test_missing_input_count_is_excluded_not_treated_as_zero(tmp_path):
    rows, base = run_and_load(tmp_path, FakeBackend(cached_ids={"What is 2 plus 2?"}))
    comp = analysis.build_comparison(rows, base)
    bad = [c for c in comp if c["id"] == "P02"][0]
    assert bad["valid"] is False and bad["input_token_ratio_bn_over_en"] is None
    s = analysis.summarize(rows, comp, base)
    assert s["valid_pairs"] == 1 and s["invalid_pairs"] == ["P02"]


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    vals = [1.5, 2.0, 2.5, 3.0, 4.0]
    assert analysis.bootstrap_ci(vals) == analysis.bootstrap_ci(vals)
    lo, hi = analysis.bootstrap_ci(vals)
    assert lo <= sum(vals) / 5 <= hi


def test_full_analysis_writes_all_outputs(tmp_path):
    rows, base = run_and_load(tmp_path, FakeBackend())
    analysis.main(results=tmp_path / "r.csv", meta=tmp_path / "m.json", out_dir=tmp_path)
    for name in ["comparison.csv", "summary.md", "summary.json", "quality_check.csv",
                 "input_token_ratio.png", "input_token_ratio_by_category.png"]:
        assert (tmp_path / name).exists(), name
