# test_groq.py — robust extraction of Groq chat reply
from groq import Groq
import os, sys

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    print("GROQ_API_KEY not set in environment. Set it and rerun.")
    sys.exit(1)

client = Groq(api_key=API_KEY)

messages = [
    {"role": "system", "content": "You are a helpful medical assistant."},
    {"role": "user", "content": "Explain diabetes in simple words."}
]

try:
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=512
    )
except Exception as e:
    print("Error calling Groq API:", e)
    sys.exit(1)

# Robust extraction of assistant content:
content = None
try:
    choice = resp.choices[0]
    msg = getattr(choice, "message", None) or choice.get("message", None)
    if msg is None:
        # try alternative fields
        content = getattr(choice, "text", None) or choice.get("text", None)
    else:
        # msg might be an object with .content or a dict with "content"
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)
except Exception:
    content = None

if not content:
    print("Couldn't parse response cleanly; printing raw response:")
    print(resp)
else:
    print("\n== Groq reply ==\n")
    print(content)
