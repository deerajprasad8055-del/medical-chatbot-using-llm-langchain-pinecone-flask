# groq_adapter.py
import os
from groq import Groq

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment")

client = Groq(api_key=API_KEY)

def chat_completion(messages, model="llama-3.1-8b-instant", max_tokens=512, **kwargs):
    """
    messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
    returns: assistant text (string)
    """
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, **kwargs)

    # robust extraction of assistant content
    content = None
    try:
        choice = resp.choices[0]
        msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
        if msg is None:
            content = getattr(choice, "text", None) or (choice.get("text") if isinstance(choice, dict) else None)
        else:
            if isinstance(msg, dict):
                content = msg.get("content")
            else:
                content = getattr(msg, "content", None)
    except Exception:
        content = None

    if content is None:
        # fallback: return full response stringified
        return str(resp)
    return content
