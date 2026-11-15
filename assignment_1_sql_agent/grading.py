"""Grade a run against the independently computed expected answer."""
import re

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_REFUSAL = re.compile(
    r"(cannot|can't|can not|unable|not (?:be )?answer|not available|no (?:such )?"
    r"(?:data|information|table|column)|does not (?:contain|have|include)|doesn't "
    r"(?:contain|have|include)|not (?:in|part of) the database|not_answerable|"
    r"don't have|do not have|no way to)", re.I)


def extract_numbers(text):
    out = []
    for m in _NUMBER.findall(str(text)):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _close(a, b):
    return abs(a - b) <= max(0.01, abs(b) * 1e-4)


def is_unanswerable_reply(answer, question=""):
    """A refusal phrase and no invented figures (numbers from the question are ok)."""
    if not _REFUSAL.search(answer):
        return False
    allowed = extract_numbers(question)
    return all(any(_close(n, a) for a in allowed) for n in extract_numbers(answer))


def is_correct(record):
    """record: a run-log record that includes 'expected'."""
    if record["status"] != "ok":
        return False
    answer, expected = record["final_answer"], record["expected"]
    if expected == "UNANSWERABLE":
        return is_unanswerable_reply(answer, record.get("question", ""))
    expected_list = expected if isinstance(expected, list) else [expected]
    for exp in expected_list:
        if isinstance(exp, (int, float)):
            if not any(_close(n, float(exp)) for n in extract_numbers(answer)):
                return False
        else:
            if not re.search(rf"\b{re.escape(str(exp))}\b", answer, re.I):
                return False
    return True


def classify_failure(record):
    """Group a failed run into one failure type."""
    status = record["status"]
    if status in ("step_cap", "budget_cap"):
        return status
    if status == "error":
        return "agent_error (exception / unusable output)"
    if record["expected"] == "UNANSWERABLE":
        return "made_up_answer_for_unanswerable"
    if not record["tool_calls"]:
        return "answered_without_tools"
    if is_unanswerable_reply(record["final_answer"], record.get("question", "")):
        return "false_refusal"
    if record.get("tool_errors"):
        return "sql_or_tool_error_not_recovered"
    return "wrong_sql_or_wrong_number"
