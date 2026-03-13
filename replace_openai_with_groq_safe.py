# replace_openai_with_groq_safe.py
# Safer replacer: scans .py files, finds openai.ChatCompletion.create(...) calls,
# extracts messages=... (literal list or variable), removes the following
# answer = resp[...] extraction, and replaces with groq_adapter.chat_completion call.
# Creates .bak backups for modified files.
#
# Run from project root while venv active:
#   python replace_openai_with_groq_safe.py

from pathlib import Path
import re

ROOT = Path(".").resolve()

def is_ignored(path: Path) -> bool:
    for part in path.parts:
        if part.startswith(".venv") or part in ("venv", ".git"):
            return True
    return False

PY_FILES = [p for p in ROOT.rglob("*.py") if not is_ignored(p)]

def make_backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        bak.write_bytes(path.read_bytes())

def find_block_end(lines, start_idx):
    """
    Given list of lines and the index where 'openai.ChatCompletion.create' appears,
    return end index (inclusive) of the create(...) call by balancing parentheses.
    """
    text = "".join(lines[start_idx:])
    idx = text.find("openai.ChatCompletion.create")
    if idx == -1:
        return start_idx
    paren_idx = text.find("(", idx)
    if paren_idx == -1:
        return start_idx
    depth = 0
    pos = paren_idx
    while pos < len(text):
        ch = text[pos]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                pre = text[:pos+1]
                end_line = pre.count("\n")  # zero-based
                return start_idx + end_line
        pos += 1
    return start_idx

def extract_messages_from_text(block_text):
    """
    Attempt to extract messages argument: either a literal list messages=[ ... ] or messages=varname
    Returns tuple (kind, value) where kind is 'literal' or 'var' or None.
    """
    mpos = block_text.find("messages")
    if mpos == -1:
        return None, None
    eq = block_text.find("=", mpos)
    if eq == -1:
        return None, None
    i = eq + 1
    while i < len(block_text) and block_text[i].isspace():
        i += 1
    if i >= len(block_text):
        return None, None
    if block_text[i] == "[":
        depth = 0
        start = i
        pos = i
        while pos < len(block_text):
            ch = block_text[pos]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = pos
                    messages_text = block_text[start:end+1]
                    return "literal", messages_text
            pos += 1
        return None, None
    else:
        var_match = re.match(r"([A-Za-z_]\w*)", block_text[i:])
        if var_match:
            varname = var_match.group(1)
            return "var", varname
    return None, None

def find_answer_extraction(lines, start_after_idx):
    """
    From start_after_idx (line after the create call), search forward a few lines
    for answer extraction patterns. Return index if found, else None.
    """
    patterns = [
        re.compile(r"^\s*answer\s*=\s*[\w\W]*choices\W*\]\W*\[0\W*].*message.*content", re.IGNORECASE),
        re.compile(r"^\s*answer\s*=\s*resp\[['\"]choices['\"]\]\[0\]\[['\"]message['\"]\]\[['\"]content['\"]\]", re.IGNORECASE),
        re.compile(r"^\s*\w+\s*=\s*resp\[['\"]choices['\"]\]\[0\]\[['\"]message['\"]\]\[['\"]content['\"]\]", re.IGNORECASE),
        re.compile(r"^\s*answer\s*=\s*resp\.choices\[0\]\.message(\.content|\.get\('content'\))", re.IGNORECASE)
    ]
    max_lookahead = 8
    for offset in range(0, max_lookahead):
        idx = start_after_idx + offset
        if idx >= len(lines):
            break
        line = lines[idx]
        for pat in patterns:
            if pat.search(line):
                return idx
    return None

def process_file(path: Path):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    changed = False
    i = 0
    new_lines = []
    while i < len(lines):
        line = lines[i]
        if "openai.ChatCompletion.create" in line:
            end_idx = find_block_end(lines, i)
            block_text = "".join(lines[i:end_idx+1])
            kind, messages = extract_messages_from_text(block_text)
            ans_idx = find_answer_extraction(lines, end_idx+1)

            repl_lines = []
            repl_lines.append("from groq_adapter import chat_completion\n\n")
            if kind == "literal" and messages:
                repl_lines.append(f"messages = {messages}\n")
                repl_lines.append("answer = chat_completion(messages)\n")
            elif kind == "var" and messages:
                repl_lines.append(f"answer = chat_completion({messages})\n")
            else:
                repl_lines.append("# NOTE: automatic conversion couldn't extract messages literal; please inspect\n")
                repl_lines.append("# Original openai.ChatCompletion.create block removed. Provide 'messages' variable.\n")
                repl_lines.append("answer = chat_completion(messages)  # ensure 'messages' exists here\n")

            new_lines.extend(repl_lines)
            if ans_idx is not None:
                i = ans_idx + 1
            else:
                i = end_idx + 1
            changed = True
            continue
        else:
            new_lines.append(line)
            i += 1

    if changed:
        print(f"Updating: {path}")
        make_backup(path)
        path.write_text("".join(new_lines), encoding="utf-8")
    return changed

def main():
    print("Scanning .py files under", ROOT)
    for p in PY_FILES:
        try:
            process_file(p)
        except Exception as e:
            print("Error processing", p, e)

if __name__ == "__main__":
    main()
