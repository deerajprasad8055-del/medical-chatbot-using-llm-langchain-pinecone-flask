# replace_openai_with_groq.py
# Usage: run from your project root with your venv active:
#    python replace_openai_with_groq.py
#
# This script:
# - searches .py files (recursively)
# - finds common patterns using OpenAI ChatCompletion
# - replaces them with equivalent calls to groq_adapter.chat_completion
# - creates backups of changed files with .bak suffix

import re
import io
import os
from pathlib import Path

ROOT = Path(".").resolve()
PY_FILES = list(ROOT.rglob("*.py"))

# Patterns to detect: (multiline)
from groq_adapter import chat_completion

messages = [...]
answer = chat_completion(messages)
# We'll do multiple targeted replacements.

def make_backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())

def replace_messages_literal_block(text):
    """
from groq_adapter import chat_completion

messages = [ ... ]
answer = chat_completion(messages)
    -> with:
      from groq_adapter import chat_completion
      messages = [ ... ]
      answer = chat_completion(messages)
    """
    pattern = re.compile(
        r"""(
            (?:\s*import\s+openai\s*\n)?                                          # optional import openai
            (?:\s*openai\.api_key\s*=\s*.*\n)?                                   # optional openai.api_key = ...
from groq_adapter import chat_completion

# NOTE: automatic conversion couldn't extract messages literal; please inspect
# Original openai.ChatCompletion.create block removed. Provide 'messages' variable.
answer = chat_completion(messages)  # ensure 'messages' exists here
            \)\s*                                                               # end create(...)
            \s*(?:\n\s*)*
            \s*(?:answer\s*=\s*(?:{resp}\["choices"\]\[0\]\["message"\]\["content"\]|{resp}\['choices'\]\[0\]\['message'\]\['content'\]|{resp}\.choices\[0\]\.message\['content'\]|{resp}\.choices\[0\]\.message\.content))
        )""".format(resp=r"(?P>resp_var)"),
        re.VERBOSE | re.DOTALL,
    )

    def repl(m):
        messages_text = m.group("messages_list")
        replacement = (
            "from groq_adapter import chat_completion\n\n"
            f"messages = {messages_text}\n"
            "answer = chat_completion(messages)\n"
        )
        return replacement

    new_text, n = pattern.subn(repl, text)
    return new_text, n

def replace_messages_variable_block(text):
    """
from groq_adapter import chat_completion

answer = chat_completion(openai)
      answer = resp[...]
    -> replace the resp + answer extraction with chat_completion(messages_var)
    """
    pattern = re.compile(
        r"""
        (?P<resp_var>\w+)\s*=\s*openai\.ChatCompletion\.create\s*\(
            (?P<args>.*?\bmessages\s*=\s*(?P<messages_var>[A-Za-z_]\w*)[^)]*)
        \)\s*
        (?:\n\s*)*
        \s*answer\s*=\s*(?:{resp}\["choices"\]\[0\]\["message"\]\["content"\]|{resp}\['choices'\]\[0\]\['message'\]\['content'\]|{resp}\.choices\[0\]\.message\['content'\]|{resp}\.choices\[0\]\.message\.content)
        """.format(resp=r"(?P>resp_var)"),
        re.VERBOSE | re.DOTALL,
    )

    def repl(m):
        messages_var = m.group("messages_var")
        replacement = f"from groq_adapter import chat_completion\n\nanswer = chat_completion({messages_var})\n"
        return replacement

    new_text, n = pattern.subn(repl, text)
    return new_text, n

def replace_simple_resp_answer(text):
    """
    Replace simple extraction patterns:
      answer = resp['choices'][0]['message']['content']
      answer = resp["choices"][0]["message"]["content"]
      answer = resp.choices[0].message.content
    when resp was created earlier by openai. This pass is conservative: we'll replace just the extraction
    with a call to chat_completion if we can find a nearby messages variable; otherwise we leave it.
    """
    # This one is too fragile to auto-infer messages; skip aggressive replacements.
    return text, 0

def remove_or_replace_imports(text):
    """
    Optionally remove import openai and openai.api_key lines if present and replace them with groq import.
    But we should only add 'from groq_adapter import chat_completion' where needed -- other replacements already insert it.
    This function removes stray 'import openai' if present and unused.
    For safety, we will not remove imports automatically unless the file now contains 'chat_completion' after previous replacements.
    """
    return text, 0

def process_file(path: Path):
    text = path.read_text(encoding="utf-8")
    original = text
    total_changes = 0

    text, n1 = replace_messages_literal_block(text)
    total_changes += n1

    text, n2 = replace_messages_variable_block(text)
    total_changes += n2

    # optionally other replacement passes here
    # text, n3 = replace_simple_resp_answer(text)
    # total_changes += n3

    if total_changes > 0 and text != original:
        print(f"Updating: {path} (replacements: {total_changes})")
        make_backup(path)
        path.write_text(text, encoding="utf-8")

def main():
    print("Scanning .py files under", ROOT)
    for p in PY_FILES:
        # skip virtualenv folders
        if any(part.startswith(".venv") or part in ("venv", ".git") for part in p.parts):
            continue
        try:
            process_file(p)
        except Exception as e:
            print("Error processing", p, e)

if __name__ == "__main__":
    main()
