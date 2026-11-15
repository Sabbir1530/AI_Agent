"""Agent loop tests with a scripted fake model (no Ollama needed)."""
import json
import pytest
from assignment_1_sql_agent.agent import run_agent, _parse_action
from assignment_1_sql_agent.create_database import build_database


@pytest.fixture
def db(tmp_path):
    return build_database(tmp_path / "t.db")


class Scripted:
    """Returns scripted replies for chat(); generate() returns scripted SQL JSON."""
    def __init__(self, replies, sql_reply='{"sql":"SELECT COUNT(*) AS answer FROM customers","explanation":"x"}',
                 tokens=50):
        self.replies, self.sql_reply, self.tokens, self.i = list(replies), sql_reply, tokens, 0

    def chat(self, messages, **kw):
        content = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        return {"message": {"content": content}, "prompt_eval_count": self.tokens, "eval_count": 5}

    def generate(self, prompt, **kw):
        return {"message": {"content": self.sql_reply}, "prompt_eval_count": 40, "eval_count": 10}


GEN = json.dumps({"tool": "generate_sql", "args": {"question": "q"}})
EXE = json.dumps({"tool": "execute_sql", "args": {"sql": "SELECT COUNT(*) AS answer FROM customers"}})
FIN = json.dumps({"final": "There are 30 customers."})


def test_happy_path_ok(db):
    log = run_agent("How many customers?", client=Scripted([GEN, EXE, FIN]), db_path=db)
    assert log["status"] == "ok" and "30" in log["final_answer"]
    assert log["tool_calls"] == ["generate_sql", "execute_sql"] and log["sql"]


def test_final_without_tool_is_pushed_back_once(db):
    log = run_agent("q", client=Scripted([FIN, GEN, EXE, FIN]), db_path=db)
    assert log["status"] == "ok" and log["tool_calls"] == ["generate_sql", "execute_sql"]


def test_step_cap(db):
    log = run_agent("q", client=Scripted([GEN]), db_path=db, max_steps=3, token_budget=10**6)
    assert log["status"] == "step_cap" and log["steps"] == 3


def test_budget_cap(db):
    log = run_agent("q", client=Scripted([GEN], tokens=900), db_path=db, max_steps=20, token_budget=1500)
    assert log["status"] == "budget_cap"


def test_sql_error_is_returned_to_model_and_retried(db):
    bad = json.dumps({"tool": "execute_sql", "args": {"sql": "SELECT nope FROM customers"}})
    log = run_agent("q", client=Scripted([GEN, bad, EXE, FIN]), db_path=db)
    assert log["status"] == "ok" and log["tool_errors"] == 1


def test_write_attempt_is_rejected_and_loop_continues(db):
    bad = json.dumps({"tool": "execute_sql", "args": {"sql": "DELETE FROM customers"}})
    log = run_agent("q", client=Scripted([GEN, bad, EXE, FIN]), db_path=db)
    assert log["status"] == "ok" and log["tool_errors"] == 1


def test_invalid_json_does_not_crash_run(db):
    log = run_agent("q", client=Scripted(["oops", GEN, EXE, FIN]), db_path=db)
    assert log["status"] == "ok" and log["parse_errors"] == 1


def test_unknown_tool_is_reported_to_model(db):
    log = run_agent("q", client=Scripted(['{"tool":"rm","args":{}}', GEN, EXE, FIN]), db_path=db)
    assert log["status"] == "ok" and log["tool_errors"] == 1


def test_calculator_tool_used(db):
    calc = json.dumps({"tool": "calculator", "args": {"expression": "(3996.5-1596)/3996.5*100"}})
    log = run_agent("q", client=Scripted([calc, FIN]), db_path=db)
    assert log["status"] == "ok" and log["tool_calls"] == ["calculator"]


def test_backend_exception_gives_error_status(db):
    class Boom(Scripted):
        def chat(self, *a, **k):
            raise RuntimeError("ollama down")
    log = run_agent("q", client=Boom([FIN]), db_path=db)
    assert log["status"] == "error" and "ollama down" in log["final_answer"]


def test_parse_action_extracts_json_from_chatter():
    assert _parse_action('Sure! {"final": "x"} done')["final"] == "x"
    with pytest.raises(ValueError):
        _parse_action("no json here")
