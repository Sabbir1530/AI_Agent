"""Small Ollama HTTP client. No hosted API is used."""

import requests

DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"


class OllamaClient:
    def __init__(self, model=DEFAULT_MODEL, base_url=DEFAULT_URL, timeout=180):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def ensure_ready(self):
        """Fail early with a clear message if Ollama or the model is missing."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}. "
                "Start it with `ollama serve`."
            ) from exc
        names = {m.get("name", "") for m in r.json().get("models", [])}
        wanted = self.model if ":" in self.model else self.model + ":latest"
        if wanted not in names:
            raise RuntimeError(
                f"Model '{self.model}' is not installed. "
                f"Run `ollama pull {self.model}`."
            )

    def chat(self, messages, temperature=0, seed=42, num_predict=512,
             keep_alive=None, tools=None, format=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": seed,
                "num_predict": num_predict,
            },
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if tools:
            payload["tools"] = tools
        if format:
            payload["format"] = format  # e.g. "json" forces valid JSON output

        response = requests.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            message = f"Ollama request failed ({response.status_code})"
            if detail:
                message += f": {detail}"
            raise requests.HTTPError(message, response=response) from exc
        return response.json()

    def generate(self, prompt, system="", temperature=0, seed=42,
                 num_predict=256, keep_alive=None, format=None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, seed=seed,
                         num_predict=num_predict, keep_alive=keep_alive,
                         format=format)
