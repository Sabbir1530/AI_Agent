from assignment_1_sql_agent.grading import is_correct, classify_failure, extract_numbers
from assignment_1_sql_agent.questions import QUESTIONS, expected_answer
from assignment_1_sql_agent.create_database import build_database


def rec(expected, answer, status="ok", tools=("generate_sql", "execute_sql"), q="", **kw):
    return {"expected": expected, "final_answer": answer, "status": status,
            "tool_calls": list(tools), "question": q, **kw}


def test_numeric_match_with_commas_and_text():
    assert is_correct(rec(119774.5, "Total revenue is $119,774.50."))
    assert not is_correct(rec(119774.5, "Total revenue is 119,000."))


def test_all_numbers_needed_for_multi_value_answers():
    assert is_correct(rec([1079.05, 30729.99], "Average 1079.05, which is 30729.99% above the minimum"))
    assert not is_correct(rec([1079.05, 30729.99], "Average 1079.05"))


def test_string_answers():
    assert is_correct(rec("Rajshahi", "Rajshahi has the most customers."))
    assert not is_correct(rec("Rajshahi", "Dhaka"))


def test_non_ok_status_never_correct():
    assert not is_correct(rec(30, "30", status="step_cap"))


def test_unanswerable_needs_refusal_and_no_invented_number():
    q = "What was the exact GDP of Bangladesh in 2025?"
    assert is_correct(rec("UNANSWERABLE", "The database cannot answer this.", q=q))
    assert is_correct(rec("UNANSWERABLE", "I cannot find GDP for 2025 in the database.", q=q))
    assert not is_correct(rec("UNANSWERABLE", "The GDP was 450 billion.", q=q))
    assert not is_correct(rec("UNANSWERABLE", "It cannot be known but about 450 billion", q=q))


def test_failure_classes():
    assert classify_failure(rec(1, "", status="step_cap")) == "step_cap"
    assert classify_failure(rec("UNANSWERABLE", "42")) == "made_up_answer_for_unanswerable"
    assert classify_failure(rec(1, "2", tools=())) == "answered_without_tools"
    assert classify_failure(rec(1, "2", tool_errors=1)) == "sql_or_tool_error_not_recovered"
    assert classify_failure(rec(1, "2")) == "wrong_sql_or_wrong_number"


def test_question_set_meets_assignment_requirements(tmp_path):
    kinds = [k for _, k, _, _ in QUESTIONS]
    assert len(QUESTIONS) >= 15
    assert kinds.count("unanswerable") >= 2 and kinds.count("multi") >= 2
    assert {"simple", "join", "grouping", "calculation"} <= set(kinds)
    db = build_database(tmp_path / "e.db")
    exp = {qid: expected_answer(sql, db) for qid, _, _, sql in QUESTIONS}
    assert exp["Q01"] == 30 and exp["Q18"] == "UNANSWERABLE"
    assert len(exp["Q16"]) == 2


def test_extract_numbers():
    assert extract_numbers("1,596.00 and -3") == [1596.0, -3.0]
