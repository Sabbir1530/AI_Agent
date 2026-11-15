"""Hand-written bounded agent loop using JSON tool requests (no framework).

Loop controls: step cap (MAX_STEPS), token budget (TOKEN_BUDGET) and a clear
final status per run: ok | step_cap | budget_cap | error.
"""
import json
import re
import time
from shared.ollama_client import DEFAULT_MODEL, OllamaClient
from shared.token_counter import usage_from_response
from assignment_1_sql_agent.tools.sql_tools import (
    SCHEMA, generate_sql_tool, execute_sql_tool
)
from assignment_1_sql_agent.tools.calculator import calculator_tool
from assignment_1_sql_agent.create_database import DB_PATH

MODEL = DEFAULT_MODEL
MAX_STEPS = 8          # model turns before status "step_cap"
TOKEN_BUDGET = 8000    # input+output tokens (incl. generate_sql) before "budget_cap"
MAX_TOOL_MESSAGE_CHARS = 1500

SYSTEM_PROMPT = f"""You are a database calculation agent.
You must use tools for every number; never guess or calculate in your head.
Available tools:
1) generate_sql: args {{ "question": "..." }}  -> returns SQL for the question
2) execute_sql: args {{ "sql": "SELECT ..." }}  -> runs the SQL, returns rows or an error
3) calculator: args {{ "expression": "(10 - 4) / 4 * 100" }}  -> exact arithmetic
Database schema:
{SCHEMA}
At each turn output ONLY JSON in one of these forms:
{{"tool":"generate_sql","args":{{"question":"..."}}}}
{{"tool":"execute_sql","args":{{"sql":"SELECT ..."}}}}
{{"tool":"calculator","args":{{"expression":"..."}}}}
{{"final":"answer based only on tool results"}}
Typical flow: generate_sql, then execute_sql with that SQL, then final.
If execute_sql returns an error, fix the SQL and try again.
If the database cannot answer the question, the final answer must say it cannot
be answered from the database. Do not invent facts."""


def _parse_action(content):
    """Parse the model reply into a dict. Raises ValueError if impossible."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("reply is not JSON")
        try:
            action = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"reply is not valid JSON ({exc.msg})") from exc
    if not isinstance(action, dict):
        raise ValueError("reply must be a JSON object")
    return action


def _clip(text):
    if len(text) <= MAX_TOOL_MESSAGE_CHARS:
        return text
    return text[:MAX_TOOL_MESSAGE_CHARS] + '... [truncated]'


def _run_tool(tool, args, question, client, db_path, log, json_mode):
    """Run one tool; always returns a JSON-able dict (errors included)."""
    if tool == "generate_sql":
        result = generate_sql_tool(client, args.get("question") or question,
                                   json_mode=json_mode)
        usage = result.pop("_usage", {})
        log["tokens"] += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if result.get("sql"):
            log["sql"].append(result["sql"])
        return result
    if tool == "execute_sql":
        sql = args.get("sql", "")
        log["sql"].append(sql)
        try:
            return execute_sql_tool(sql, db_path)
        except Exception as exc:  # returned to the model so it can repair it
            return {"error": str(exc)}
    if tool == "calculator":
        try:
            return calculator_tool(args.get("expression", ""))
        except Exception as exc:
            return {"error": str(exc)}
    return {"error": f"Unknown tool '{tool}'. Use generate_sql, execute_sql or calculator."}


def run_agent(question, client=None, db_path=DB_PATH, max_steps=MAX_STEPS,
              token_budget=TOKEN_BUDGET, json_mode=True):
    client = client or OllamaClient(model=MODEL)
    started = time.perf_counter()
    log = {"question": question, "tool_calls": [], "sql": [],
           "final_answer": "", "status": "error", "tokens": 0,
           "time_s": 0.0, "steps": 0, "tool_errors": 0, "parse_errors": 0}
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    nudged = False
    try:
        for step in range(max_steps):
            if log["tokens"] >= token_budget:
                break
            log["steps"] = step + 1
            response = client.chat(messages, temperature=0, seed=42,
                                   num_predict=350,
                                   format="json" if json_mode else None)
            log["tokens"] += usage_from_response(response)["total_tokens"]
            content = response["message"]["content"]
            messages.append({"role": "assistant", "content": content})
            try:
                action = _parse_action(content)
            except ValueError as exc:
                log["parse_errors"] += 1
                messages.append({"role": "user", "content":
                    f"Invalid reply: {exc}. Reply with ONLY one JSON object."})
                continue

            if "final" in action:
                if not log["tool_calls"] and not nudged:
                    # Numbers must come from tools: push back once.
                    nudged = True
                    messages.append({"role": "user", "content":
                        "You have not used any tool yet. Call generate_sql first "
                        "(it will report if the database cannot answer)."})
                    continue
                log["final_answer"] = str(action["final"])
                log["status"] = "ok"
                break

            tool = action.get("tool")
            args = action.get("args")
            args = args if isinstance(args, dict) else {}
            log["tool_calls"].append(str(tool))
            result = _run_tool(tool, args, question, client, db_path, log, json_mode)
            if "error" in result:
                log["tool_errors"] += 1
            messages.append({"role": "user", "content":
                "Tool result: " + _clip(json.dumps(result, default=str))
                + "\nContinue. Use another tool if needed, otherwise return final JSON."})
        else:
            log["status"] = "step_cap"  # loop used every step without a final
        if log["status"] != "ok":
            # Reached via the step cap or the budget `break` above.
            if log["tokens"] >= token_budget:
                log["status"] = "budget_cap"
            elif log["status"] != "step_cap":
                log["status"] = "step_cap"
            log["final_answer"] = (
                "Stopped: token budget exceeded." if log["status"] == "budget_cap"
                else "Stopped: maximum agent steps reached.")
    except Exception as exc:  # returned to the model so it can repair it
            return {"error": str(exc)}
    if tool == "calculator":
        try:
            return calculator_tool(args.get("expression", ""))
        except Exception as exc:
            return {"error": str(exc)}
    return {"error": f"Unknown tool '{tool}'. Use generate_sql, execute_sql or calculator."}


def run_agent(question, client=None, db_path=DB_PATH, max_steps=MAX_STEPS,
              token_budget=TOKEN_BUDGET, json_mode=True):
    client = client or OllamaClient(model=MODEL)
    started = time.perf_counter()
    log = {"question": question, "tool_calls": [], "sql": [],
           "final_answer": "", "status": "error", "tokens": 0,
           "time_s": 0.0, "steps": 0, "tool_errors": 0, "parse_errors": 0}
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    nudged = False
    try:
        for step in range(max_steps):
            if log["tokens"] >= token_budget:
                break
            log["steps"] = step + 1
            response = client.chat(messages, temperature=0, seed=42,
                                   num_predict=350,
                                   format="json" if json_mode else None)
            log["tokens"] += usage_from_response(response)["total_tokens"]
            content = response["message"]["content"]
            messages.append({"role": "assistant", "content": content})
            try:
                action = _parse_action(content)
            except ValueError as exc:
                log["parse_errors"] += 1
                messages.append({"role": "user", "content":
                    f"Invalid reply: {exc}. Reply with ONLY one JSON object."})
                continue

            if "final" in action:
                if not log["tool_calls"] and not nudged:
                    # Numbers must come from tools: push back once.
                    nudged = True
                    messages.append({"role": "user", "content":
                        "You have not used any tool yet. Call generate_sql first "
                        "(it will report if the database cannot answer)."})
                    continue
                log["final_answer"] = str(action["final"])
                log["status"] = "ok"
                break

            tool = action.get("tool")
            args = action.get("args")
            args = args if isinstance(args, dict) else {}
            log["tool_calls"].append(str(tool))
            result = _run_tool(tool, args, question, client, db_path, log, json_mode)
            if "error" in result:
                log["tool_errors"] += 1
            messages.append({"role": "user", "content":
                "Tool result: " + _clip(json.dumps(result, default=str))
                + "\nContinue. Use another tool if needed, otherwise return final JSON."})
        else:
            log["status"] = ("budget_cap" if log["tokens"] >= token_budget
                             else "step_cap")
        if log["status"] == "error" and not log["final_answer"]:
            # loop ended via `break` on the budget check or fell through
            log["status"] = ("budget_cap" if log["tokens"] >= token_budget
                             else "step_cap")
            log["final_answer"] = (
                "Stopped: token budget exceeded." if log["status"] == "budget_cap"
                else "Stopped: maximum agent steps reached.")
        elif log["status"] in ("step_cap", "budget_cap"):
            log["final_answer"] = (
                "Stopped: token budget exceeded." if log["status"] == "budget_cap"
                else "Stopped: maximum agent steps reached.")
    except Exception as exc:
        log["status"] = "error"
        log["final_answer"] = f"Error: {exc}"
    log["time_s"] = round(time.perf_counter() - started, 4)
    return log


if __name__ == "__main__":
    import sys
    if not DB_PATH.exists():
        from assignment_1_sql_agent.create_database import build_database
        build_database()
    question = " ".join(sys.argv[1:]) or "How many customers are there?"
    print(json.dumps(run_agent(question), indent=2))
