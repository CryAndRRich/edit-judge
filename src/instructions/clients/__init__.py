"""Model clients for the request analysis. Register each as name -> "module:Class", e.g.
    CLIENTS = {"qwen2.5-vl-7b": ".qwen_vl:QwenVL", "gemini-flash": ".gemini:Gemini"}
API keys are read from .env (or Kaggle secrets passed as environment variables), never written in code."""

CLIENTS: dict[str, str] = {}
