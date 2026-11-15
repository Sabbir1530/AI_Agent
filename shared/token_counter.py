"""Token/cost helpers based on Ollama's own usage counters.

Ollama reports prompt_eval_count (input tokens) and eval_count (output
tokens) in every response. Notes on how they are handled:

* Template/system overhead: the input count includes the chat template and
  the system prompt. Use ``measure_baseline`` once to measure that fixed
  overhead with a 1-character prompt; every call also uses the same fixed
  system prompt, so the overhead is identical for both languages.
* Prompt caching: Ollama may reuse a cached prompt, which can shrink (or
  even omit) prompt_eval_count. The runner uses keep_alive=0 so the model
  is unloaded after each call, and ``input_count_ok`` flags any response
  whose input count is missing or zero so it can be excluded, not silently
  treated as a real measurement.
"""
INPUT_PRICE_PER_MILLION = 0.15
OUTPUT_PRICE_PER_MILLION = 0.60


def usage_from_response(response):
    input_tokens = int(response.get("prompt_eval_count") or 0)
    output_tokens = int(response.get("eval_count") or 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def input_count_ok(response):
    """True if Ollama reported a usable input-token count (not cached/missing)."""
    return int(response.get("prompt_eval_count") or 0) > 0


def was_truncated(response):
    """True if generation stopped because num_predict was reached."""
    return response.get("done_reason") == "length"


def calculate_cost(input_tokens, output_tokens):
    return (
        input_tokens * INPUT_PRICE_PER_MILLION
        + output_tokens * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000


def measure_baseline(client, system, probe=".", **options):
    """Input tokens used by the template + system prompt + a 1-char user prompt."""
    response = client.generate(probe, system=system, **options)
    return usage_from_response(response)["input_tokens"]
