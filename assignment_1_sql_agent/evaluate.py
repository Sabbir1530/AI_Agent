"""Run the questions N times each and grade against independent SQL.

Usage (from the project root):
    python -m assignment_1_sql_agent.evaluate                # 3 trials
    python -m assignment_1_sql_agent.evaluate --out run_log_before.jsonl --no-json-mode
"""
import argparse
import json
from pathlib import Path
from assignment_1_sql_agent.agent import run_agent, MODEL
from assignment_1_sql_agent.create_database import DB_PATH, build_database
from assignment_1_sql_agent.grading import is_correct
from assignment_1_sql_agent.questions import QUESTIONS, expected_answer
from shared.ollama_client import OllamaClient

HERE = Path(__file__).parent


def main(argv=None, client=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "run_log.jsonl"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--only", default="", help="comma-separated ids, e.g. Q01,Q05")
    ap.add_argument("--no-json-mode", action="store_true",
                    help="do not force Ollama JSON output (for before/after runs)")
    args = ap.parse_args(argv)

    if not DB_PATH.exists():
        build_database()
    client = client or OllamaClient(model=args.model)
    if hasattr(client, "ensure_ready"):
        client.ensure_ready()

    chosen = {q.strip() for q in args.only.split(",") if q.strip()}
    questions = [q for q in QUESTIONS if not chosen or q[0] in chosen]
    expected = {qid: expected_answer(sql, DB_PATH) for qid, _, _, sql in questions}

    # The question set with expected answers (a submission item)
    qfile = HERE / "questions_expected.json"
    qfile.write_text(json.dumps(
        [{"id": qid, "kind": kind, "question": q, "expected": expected[qid]}
         for qid, kind, q, _ in questions], indent=2, ensure_ascii=False),
        encoding="utf-8")

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for qid, kind, question, _ in questions:
            for trial in range(1, args.trials + 1):
                result = run_agent(question, client=client,
                                   json_mode=not args.no_json_mode)
                record = {"id": qid, "kind": kind, "trial": trial,
                          "model": args.model, "expected": expected[qid],
                          **result}
                record["correct"] = is_correct(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(qid, trial, result["status"],
                      "PASS" if record["correct"] else "FAIL",
                      result["final_answer"][:80])
    print(f"Saved: {out}\nSaved: {qfile}")


if __name__ == "__main__":
    main()
