# react_agent_fixed.py
"""
Minimal fixed version:
- Uses OpenAI 1.0+ client (OpenAI())
- Avoids fragile LangGraph wiring; falls back to internal runner
"""

import os
import time
import json
import requests
import ast
import logging

# modern OpenAI client
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY environment variable and re-run.")

# optional: attempt to import langgraph. We'll not force its usage yet.
try:
    import langgraph  # type: ignore
    HAS_LANGGRAPH = True
    logger.info("langgraph detected (will not auto-use to avoid API mismatch).")
except Exception:
    HAS_LANGGRAPH = False
    logger.info("langgraph not found; using built-in runner.")

# -------------------------
# Tools (same safe implementations)
# -------------------------
def tool_search(query: str, max_chars: int = 800) -> str:
    if not query:
        return "search: empty query"
    headers = {"User-Agent": "ReactAgent/1.0 (educational project; contact@example.com)"}
    params = {"action": "query", "format": "json", "list": "search", "srsearch": query, "srlimit": 1}
    resp = requests.get("https://en.wikipedia.org/w/api.php", params=params, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return f"No wiki result for: {query}"
    title = hits[0]["title"]
    summary_resp = requests.get(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.requote_uri(title)}",
        headers=headers,
        timeout=10,
    )
    if summary_resp.status_code != 200:
        return f"Found page {title} but could not fetch summary."
    summary = summary_resp.json().get("extract", "")
    return f"{title}: {summary[:max_chars]}"

ALLOWED_AST_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub,
    ast.Load, ast.FloorDiv
}

def safe_eval(expr: str) -> str:
    try:
        tree = ast.parse(expr, mode="eval")
    except Exception as e:
        return f"calc error: invalid expression ({e})"
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_AST_NODES:
            return f"calc error: disallowed expression part {type(node).__name__}"
    try:
        result = eval(compile(tree, filename="<ast>", mode="eval"), {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"calc error: evaluation failed ({e})"

def dispatch_function_call(function_name: str, args: dict) -> str:
    if function_name == "search":
        return tool_search(args.get("query", ""))
    if function_name == "calc":
        return safe_eval(args.get("expression", ""))
    return f"unknown function {function_name}"

FUNCTIONS = [
    {
        "name": "search",
        "description": "Search Wikipedia and return a short summary for a query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for on Wikipedia"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calc",
        "description": "Evaluate a mathematical expression and return the numeric result",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "A math expression to evaluate, e.g. (3+4)*5"}
            },
            "required": ["expression"]
        }
    }
]

SYSTEM_PROMPT = """You are an assistant that follows the ReAct pattern.
When you need facts or calculations, call the appropriate function:
- use search(query) to look up concise factual summaries.
- use calc(expression) to compute numeric results.
When you know the final response for the user, reply with "Final Answer: <answer>".
Be concise and avoid calling tools unnecessarily.
"""

# -------------------------
# Use modern client for model calls
# -------------------------
def call_model_with_functions(messages, model="gpt-4-0613", functions=FUNCTIONS, max_tokens=800):
    """
    Use the OpenAI 1.0+ client interface.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call="auto",
        temperature=0.0,
        max_tokens=max_tokens
    )
    return resp

# -------------------------
# Pure-Python fallback runner (working)
# -------------------------
def run_react_agent_fallback(user_question: str, max_steps: int = 8, model: str = "gpt-4-0613"):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_question}]
    step = 0
    while step < max_steps:
        step += 1
        resp = call_model_with_functions(messages, model=model)
        choice = resp.choices[0]
        msg = choice.message

        # assistant direct textual reply case
        if msg.role == "assistant" and msg.content:
            assistant_text = msg.content
            logger.info("Assistant: %s", assistant_text)
            messages.append({"role": "assistant", "content": assistant_text})
            if "Final Answer:" in assistant_text:
                return assistant_text.split("Final Answer:")[-1].strip()
            continue

        # function call case
        if msg.function_call:
            fname = msg.function_call.name
            argstr = msg.function_call.arguments or "{}"
            try:
                args = json.loads(argstr)
            except Exception:
                args = {}
            logger.info("Calling tool %s with %s", fname, args)
            # Add the assistant's function call message to maintain conversation context
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "function_call": {"name": fname, "arguments": argstr}
            })
            result = dispatch_function_call(fname, args)
            messages.append({"role": "function", "name": fname, "content": result})
            time.sleep(0.12)
            continue

    return "Agent stopped: reached max steps or no final answer."

# -------------------------
# LangGraph path: disabled-safe fallback
# -------------------------
def build_and_run_langgraph(question: str, max_steps: int = 8):
    """
    For now we avoid invoking StateGraph directly because local langgraph
    APIs differ across installs. If you want proper LangGraph wiring, I can
    generate it once you paste the output of:

        import langgraph, inspect
        print(langgraph)
        print(inspect.getsource(langgraph.graph.state.StateGraph)[:800])

    For now return fallback so the script runs reliably.
    """
    if HAS_LANGGRAPH:
        logger.warning("LangGraph detected but integration is disabled to avoid API mismatch. Using fallback runner.")
    return run_react_agent_fallback(question, max_steps=max_steps)

# -------------------------
# Public runner (keeps same signature)
# -------------------------
def run(question: str, use_langgraph: bool = True):
    if use_langgraph and HAS_LANGGRAPH:
        # we deliberately avoid fragile graph invocation; fallback instead
        return build_and_run_langgraph(question)
    return run_react_agent_fallback(question)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    q = input("Question (enter for sample): ").strip()
    if not q:
        q = "When was the first public demonstration of the World Wide Web? Also compute 1991 - 1989."
    print("Running...")
    out = run(q, use_langgraph=True)
    print("\n=== Result ===")
    print(out)

