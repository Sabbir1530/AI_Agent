# Part 1: The Model as a Measured Component

Two assignments, both using a local model through Ollama (no hosted API).

| # | Folder | What it is |
|---|---|---|
| 1 | `assignment_1_sql_agent/` | An agent that answers calculation questions by calling tools against a seeded SQLite database |
| 2 | `assignment_2_token_cost/` | 50 aligned English/Bangla prompts compared on tokens, cost and latency |

**Model:** `qwen2.5:3b` served by Ollama (change with `--model`; always say which tag you used in your report).
Prices are notional: $0.15 per million input tokens, $0.60 per million output tokens.

## Setup (once)

```bash
pip install -r requirements.txt          # requests, matplotlib, pytest
ollama pull qwen2.5:3b                   # then keep `ollama serve` running
python -m pytest                         # 60+ tests, no Ollama needed (fake backends)
```

All commands below are run from this folder (the project root).

## Assignment 1: run it (4 commands)

```bash
python -m assignment_1_sql_agent.create_database        # rebuild the DB (seed 20261111)
python -m assignment_1_sql_agent.agent "How many customers are there?"   # one question
python -m assignment_1_sql_agent.evaluate               # 19 questions x 3 trials -> run_log.jsonl
python -m assignment_1_sql_agent.report                 # pass@1, pass^3, failures -> results_summary.md
```

Before/after for the report: run a "before" log with
`python -m assignment_1_sql_agent.evaluate --no-json-mode --out assignment_1_sql_agent/run_log_before.jsonl`,
then `python -m assignment_1_sql_agent.report --before assignment_1_sql_agent/run_log_before.jsonl`.
(`--no-json-mode` turns off Ollama's forced-JSON output, which is one of the improvements made.)

### Architecture

```
question -> agent loop (agent.py, hand-written, JSON tool requests)
              |-- generate_sql   -> separate model call -> {"sql": ...}   (tools/sql_tools.py)
              |-- execute_sql    -> read-only SQLite (mode=ro + authorizer, SELECT only, row limit)
              |-- calculator     -> safe AST arithmetic (bonus tool)      (tools/calculator.py)
              '-- final          -> answer built from tool results
caps: MAX_STEPS=8, TOKEN_BUDGET=8000 (includes the generate_sql calls)
status per run: ok | step_cap | budget_cap | error
```

**Design choice (requirement 2):** `generate_sql` is a tool that makes its *own* model call, with a focused prompt, strict rules and JSON output. The schema is also in the agent prompt so it can reason about it. This keeps SQL writing separate from the agent's decisions and lets it be tested alone.

**Errors:** SQL errors, rejected statements, unknown tools and unparseable replies are all returned to the model as text so it can retry within the step cap. A `final` answer before any tool call is pushed back once ("numbers must come from tools").

### Files

- `create_database.py` – builds `database/agent.db` (4 tables, 180 orders / 451 order_items) from a fixed seed
- `questions.py` – 19 questions (simple, join, grouping, calculation, 2 multi-step, 2 unanswerable) with **independent** SQL for expected answers (plain `sqlite3`, not the agent's tools)
- `questions_expected.json` – the question set with expected answers
- `grading.py` – correctness check (numbers with tolerance, strings, refusal for unanswerable) and failure classification
- `evaluate.py` / `report.py` – run 3 trials each and compute **pass@1** (share of all runs correct) and **pass^3** (share of questions correct in all 3 trials)
- Outputs after you run it: `run_log.jsonl` (question, tool calls, SQL, final answer, status, tokens, time per run), `results_summary.md`

## Assignment 2: run it (3 commands)

```bash
python -m assignment_2_token_cost.runner      # 100 calls -> results.csv, run_meta.json (add --resume after an interruption)
python -m assignment_2_token_cost.analysis    # comparison.csv, summary.md/json, 2 charts, quality_check.csv
python -m pytest tests/test_assignment2.py    # dataset + counting/cost tests with a fake backend
```

The runner is slow on purpose: `keep_alive=0` unloads the model after every call, so no prompt cache is reused.

### Method

- **Dataset:** `dataset.csv` (`id, category, english, bangla`) – 50 pairs in 6 categories (question, instruction, explanation, arithmetic, summarise, long 3-5 sentences). Digits are ASCII in both languages so that digit script does not distort the comparison.
- **Same settings for both languages:** model, one fixed system prompt (`Answer clearly and accurately. Be concise but complete.`), temperature 0, seed 42, `num_predict=160`. No language instruction is given for either language.
- **Token counting:** `shared/token_counter.py` uses Ollama's `prompt_eval_count` / `eval_count`.
  - *Template/system overhead:* measured once with a 1-character prompt and saved as `baseline_input_tokens` in `run_meta.json`. `comparison.csv` has both the raw ratio and a `content_only_ratio` with that baseline removed.
  - *Caching:* fresh state via `keep_alive=0`; any response with a missing/zero input count is flagged (`input_count_ok=False`) and that pair is excluded from the statistics, never counted as zero.
  - *Truncation:* `done_reason == "length"` is recorded in `truncated`; the summary counts answers cut by `num_predict`.
- **Outputs:** `results.csv`, `comparison.csv` (per pair), `summary.md` (totals, cost ratio, mean/median/min/max/95% bootstrap CI, latency, per-category table), `input_token_ratio.png`, `input_token_ratio_by_category.png`.

## What you still must do yourself

The model was not available where this code was written, so **no results or numbers are included**. After running the commands:

1. **Review the Bangla.** I wrote the Bangla side of `dataset.csv`; the assignment asks you to write or carefully check it. Read all 50 pairs for same meaning (and say in your report that you reviewed them).
2. **Quality check:** open `assignment_2_token_cost/quality_check.csv` and fill in the verdict for the 10 pairs.
3. **Write the two 1-2 page reports** from `results_summary.md` and `summary.md`, including the model tag, the three most common failure types with examples, the before/after change, the category with the largest gap, and one surprising prompt (the summary names the lowest and highest ratio pairs to start from).
4. **Tune `TOKEN_BUDGET` / `MAX_STEPS`** in `agent.py` if many runs end in `budget_cap`; the log shows tokens per run.
5. **Fill in `AI_USE_LOG.md`** (the course requires a short log of AI-assistant use).

## Tests

`python -m pytest` runs everything with fake backends: read-only safety (DELETE/DROP/UPDATE/INSERT/multi-statement/PRAGMA rejected, row limit), calculator, agent loop (step cap, budget cap, error recovery, bad JSON), grading, Ollama client payload, token/cost arithmetic, dataset shape, runner settings, resume, and the analysis outputs.
