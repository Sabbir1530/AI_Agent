"""Run 50 aligned English/Bangla prompts and record Ollama token usage.

Identical settings for both languages: same model, same fixed system prompt,
temperature 0, seed 42, num_predict 160. keep_alive=0 unloads the model after
each call so no prompt cache is reused. Outputs: results.csv and run_meta.json
(model, settings, measured template/system baseline).
"""
import argparse
import csv
import json
import time
from pathlib import Path
from shared.ollama_client import DEFAULT_MODEL, OllamaClient
from shared.token_counter import (
    usage_from_response, calculate_cost, input_count_ok, was_truncated,
    measure_baseline,
)

HERE = Path(__file__).parent
DATASET = HERE / "dataset.csv"
RESULTS = HERE / "results.csv"
META = HERE / "run_meta.json"
SYSTEM_PROMPT = "Answer clearly and accurately. Be concise but complete."
SETTINGS = {"temperature": 0, "seed": 42, "num_predict": 160, "keep_alive": 0}
FIELDS = ["id", "category", "language", "prompt", "output",
          "input_tokens", "output_tokens", "cost_usd", "latency_s",
          "done_reason", "truncated", "input_count_ok"]
LANGUAGES = ("english", "bangla")


def _call_with_retry(client, prompt, retries=2):
    for attempt in range(retries + 1):
        try:
            start = time.perf_counter()
            response = client.generate(prompt, system=SYSTEM_PROMPT, **SETTINGS)
            return response, time.perf_counter() - start
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2)


def load_dataset(path=DATASET):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for need in ("id", "category", "english", "bangla"):
        if rows and need not in rows[0]:
            raise ValueError(f"dataset is missing column '{need}'")
    return rows


def make_row(item, language, response, latency):
    usage = usage_from_response(response)
    return {
        "id": item["id"], "category": item["category"], "language": language,
        "prompt": item[language], "output": response["message"]["content"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "cost_usd": f'{calculate_cost(usage["input_tokens"], usage["output_tokens"]):.10f}',
        "latency_s": f"{latency:.4f}",
        "done_reason": response.get("done_reason", ""),
        "truncated": was_truncated(response),
        "input_count_ok": input_count_ok(response),
    }


def run(client=None, dataset=DATASET, results=RESULTS, meta=META, resume=False,
        model=DEFAULT_MODEL, verbose=True):
    client = client or OllamaClient(model=model)
    if hasattr(client, "ensure_ready"):
        client.ensure_ready()
    prompts = load_dataset(dataset)
    results, meta = Path(results), Path(meta)

    done = set()
    if resume and results.exists():
        with results.open(encoding="utf-8", newline="") as f:
            done = {(r["id"], r["language"]) for r in csv.DictReader(f)}
    mode = "a" if done else "w"

    # Baseline: tokens added by chat template + system prompt (+1-char prompt).
    baseline = measure_baseline(client, SYSTEM_PROMPT, **{**SETTINGS, "num_predict": 1})
    meta.write_text(json.dumps({
        "model": getattr(client, "model", model), "system_prompt": SYSTEM_PROMPT,
        "settings": SETTINGS, "baseline_input_tokens": baseline,
        "baseline_note": "input tokens for template + system prompt + a 1-character user prompt",
    }, indent=2), encoding="utf-8")

    with results.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        for item in prompts:
            for language in LANGUAGES:
                if (item["id"], language) in done:
                    continue
                response, latency = _call_with_retry(client, item[language])
                row = make_row(item, language, response, latency)
                writer.writerow(row)
                f.flush()
                if verbose:
                    print(item["id"], language, row["input_tokens"],
                          row["output_tokens"], f"{latency:.2f}s")
    if verbose:
        print(f"Saved {results} (baseline input tokens = {baseline})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    run(resume=a.resume, model=a.model)
