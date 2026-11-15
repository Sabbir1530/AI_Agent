from shared.token_counter import (
    usage_from_response, calculate_cost, input_count_ok, was_truncated,
    measure_baseline,
)


def test_usage_counts_from_ollama_response():
    fake = {"prompt_eval_count": 100, "eval_count": 50}
    assert usage_from_response(fake) == {
        "input_tokens": 100, "output_tokens": 50, "total_tokens": 150}


def test_missing_counts_default_to_zero():
    assert usage_from_response({})["total_tokens"] == 0


def test_notional_cost():
    assert calculate_cost(1_000_000, 0) == 0.15
    assert calculate_cost(0, 1_000_000) == 0.60
    assert abs(calculate_cost(1000, 500) - 0.00045) < 1e-12


def test_cached_or_missing_input_count_is_flagged():
    assert input_count_ok({"prompt_eval_count": 12})
    assert not input_count_ok({"eval_count": 5})
    assert not input_count_ok({"prompt_eval_count": 0})


def test_truncation_flag():
    assert was_truncated({"done_reason": "length"})
    assert not was_truncated({"done_reason": "stop"})


def test_measure_baseline_uses_fake_backend():
    class Fake:
        def generate(self, prompt, system="", **kw):
            assert prompt == "." and system == "sys"
            return {"prompt_eval_count": 23, "eval_count": 1}
    assert measure_baseline(Fake(), "sys") == 23
