"""
Tool registry for the MLX chat app.

Add a tool by writing a function and decorating it with @tool(schema). The
schema is the standard OpenAI/JSON-schema function definition the model sees.
app.py imports TOOL_SCHEMAS (to show the model what's available), run_tool (to
execute a call), and extract_tool_calls (to find calls in the model's output).

    @tool({
        "type": "function",
        "function": {
            "name": "my_tool",
            "description": "what it does",
            "parameters": {"type": "object", "properties": {...}, "required": [...]},
        },
    })
    def my_tool(arg1, arg2=None):
        ...
        return "string result"   # always return a string
"""

import ast
from datetime import datetime, timezone
import json
import operator
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx

# Load .env (project dir) into the environment without overwriting real env vars.
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# What the model emits around a call:  <tool_call>{"name":...,"arguments":{...}}</tool_call>
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
_CALL_RE = re.compile(re.escape(TOOL_CALL_START) + r"\s*(.*?)\s*" + re.escape(TOOL_CALL_END), re.DOTALL)

# ---- registry --------------------------------------------------------------

TOOL_SCHEMAS = []          # list of schema dicts, passed to apply_chat_template(tools=...)
_REGISTRY = {}             # name -> python callable


def tool(schema):
    """Decorator: register a function and its schema as a callable tool."""
    name = schema["function"]["name"]

    def deco(fn):
        _REGISTRY[name] = fn
        TOOL_SCHEMAS.append(schema)
        return fn

    return deco


def run_tool(name, arguments):
    """Execute a registered tool. Returns a string (errors are returned, not raised,
    so the model can read and recover from them)."""
    fn = _REGISTRY.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'."
    try:
        result = fn(**(arguments or {}))
    except Exception as e:
        return f"Error running '{name}': {e}"
    return result if isinstance(result, str) else json.dumps(result)


def extract_tool_calls(text):
    """Pull every <tool_call>...</tool_call> out of model output.
    Returns a list of {"name": str, "arguments": dict}."""
    calls = []
    for blob in _CALL_RE.findall(text):
        try:
            obj = json.loads(blob.strip())
            calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
        except Exception:
            continue
    return calls


# ---- tools -----------------------------------------------------------------

@tool({
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information and return a short "
                       "summary plus the top source links.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {"type": "integer", "description": "How many sources (1-5).", "default": 3},
            },
            "required": ["query"],
        },
    },
})
def web_search(query, max_results=3):
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return "Error: TAVILY_API_KEY is not set (add it to .env)."
    max_results = max(1, min(int(max_results or 3), 5))
    r = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": max_results,
              "include_answer": True, "search_depth": "basic"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    lines = []
    if data.get("answer"):
        lines.append(f"Summary: {data['answer']}")
    for i, res in enumerate(data.get("results", []), 1):
        snippet = (res.get("content") or "").strip().replace("\n", " ")[:300]
        lines.append(f"[{i}] {res.get('title')} ({res.get('url')})\n{snippet}")
    return "\n\n".join(lines) if lines else "No results found."


# Safe arithmetic evaluator — supports + - * / ** % and parentheses only.
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


@tool({
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a basic arithmetic expression (+, -, *, /, **, %).",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. '2 * (3 + 4)'"}},
            "required": ["expression"],
        },
    },
})
def calculate(expression):
    value = _eval_node(ast.parse(expression, mode="eval").body)
    return f"{expression} = {value}"


PROJECT_ROOT = Path(__file__).parent.resolve()
SKIP_DIRS = {".venv", "__pycache__", ".git", ".cache"}
TEXT_EXTS = {
    ".py", ".html", ".css", ".js", ".json", ".md", ".txt", ".toml", ".yaml",
    ".yml", ".env", ".gitignore", ".jinja",
}


def _safe_path(path="."):
    target = (PROJECT_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        target.relative_to(PROJECT_ROOT)
    except ValueError:
        raise ValueError("path is outside the project")
    return target


def _is_skipped(path):
    return any(part in SKIP_DIRS for part in path.relative_to(PROJECT_ROOT).parts)


def _is_text_file(path):
    return path.name in {".env", ".gitignore"} or path.suffix.lower() in TEXT_EXTS


@tool({
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files and folders inside this project.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Project-relative directory.", "default": "."},
                "max_items": {"type": "integer", "description": "Maximum items to return.", "default": 80},
            },
        },
    },
})
def list_files(directory=".", max_items=80):
    try:
        base = _safe_path(directory)
    except ValueError as e:
        return f"Error: {e}"
    if not base.exists():
        return f"Error: path does not exist: {directory}"
    if not base.is_dir():
        return f"Error: not a directory: {directory}"

    max_items = max(1, min(int(max_items or 80), 200))
    lines = []
    for path in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if _is_skipped(path):
            continue
        rel = path.relative_to(PROJECT_ROOT)
        lines.append(f"{rel}/" if path.is_dir() else str(rel))
        if len(lines) >= max_items:
            lines.append("... truncated")
            break
    return "\n".join(lines) if lines else "No files found."


@tool({
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a text file from this project.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Project-relative file path."},
                "max_chars": {"type": "integer", "description": "Maximum characters to return.", "default": 12000},
            },
            "required": ["path"],
        },
    },
})
def read_file(path, max_chars=12000):
    try:
        target = _safe_path(path)
    except ValueError as e:
        return f"Error: {e}"
    if not target.exists():
        return f"Error: path does not exist: {path}"
    if not target.is_file():
        return f"Error: not a file: {path}"
    if _is_skipped(target) or not _is_text_file(target):
        return f"Error: refusing to read non-project text file: {path}"

    max_chars = max(500, min(int(max_chars or 12000), 50000))
    text = target.read_text(errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n... truncated"
    return text


@tool({
    "type": "function",
    "function": {
        "name": "search_files",
        "description": "Search project text files for a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to search for."},
                "directory": {"type": "string", "description": "Project-relative directory.", "default": "."},
                "max_results": {"type": "integer", "description": "Maximum matching lines.", "default": 30},
            },
            "required": ["query"],
        },
    },
})
def search_files(query, directory=".", max_results=30):
    if not query:
        return "Error: query is required."
    try:
        base = _safe_path(directory)
    except ValueError as e:
        return f"Error: {e}"
    if not base.exists() or not base.is_dir():
        return f"Error: not a directory: {directory}"

    max_results = max(1, min(int(max_results or 30), 100))
    matches = []
    for path in sorted(base.rglob("*")):
        if len(matches) >= max_results:
            break
        if not path.is_file() or _is_skipped(path) or not _is_text_file(path):
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for lineno, line in enumerate(lines, 1):
            if query.lower() in line.lower():
                rel = path.relative_to(PROJECT_ROOT)
                matches.append(f"{rel}:{lineno}: {line.strip()}")
                if len(matches) >= max_results:
                    break
    if len(matches) >= max_results:
        matches.append("... truncated")
    return "\n".join(matches) if matches else "No matches found."


@tool({
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current local and UTC time.",
        "parameters": {"type": "object", "properties": {}},
    },
})
def get_time():
    local_now = datetime.now().astimezone()
    utc_now = datetime.now(timezone.utc)
    return (
        f"Local time: {local_now.isoformat(timespec='seconds')}\n"
        f"UTC time: {utc_now.isoformat(timespec='seconds')}"
    )


@tool({
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": "Fetch a specific HTTP/HTTPS URL and return readable text.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
                "max_chars": {"type": "integer", "description": "Maximum characters to return.", "default": 6000},
            },
            "required": ["url"],
        },
    },
})
def web_fetch(url, max_chars=6000):
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "Error: only http and https URLs are allowed."

    max_chars = max(500, min(int(max_chars or 6000), 20000))
    r = httpx.get(url, timeout=20, follow_redirects=True)
    r.raise_for_status()

    content_type = r.headers.get("content-type", "")
    text = r.text
    if "html" in content_type:
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n... truncated"
    return text or "No readable text found."


# ---- add more tools below --------------------------------------------------
# Write a function, decorate it with @tool(schema), and it's live — app.py
# picks it up automatically via TOOL_SCHEMAS / run_tool.
