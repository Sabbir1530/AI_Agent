import pytest
from assignment_1_sql_agent.tools.calculator import calculator_tool


def test_basic_math():
    assert calculator_tool("(10 - 4) / 4 * 100")["result"] == 150
    assert calculator_tool("1,000 + 2")["result"] == 1002


@pytest.mark.parametrize("expr", ["__import__('os')", "1/0", "abs(3)", "", "2**1000", "a+1"])
def test_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ValueError):
        calculator_tool(expr)
