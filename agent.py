import argparse
import ast
import operator
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic

MODEL = "claude-opus-4-7"
SYSTEM_PROMPT = (
    "You are a helpful file system assistant. You have read-only tools to explore a directory, "
    "read files, search for files by name, get the current time, and do math. "
    "Use the tools when they help answer the user's question. "
    "When showing file contents, summarize unless the user asks for the raw text. "
    "Respond concisely in the user's language."
)

TOOLS = [
    {
        "name": "list_directory",
        "description": "List files and subdirectories in a given path (relative to the workspace root).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path, e.g. '.' or 'src'"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a text file (relative to the workspace root). Max 10KB returned.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_files",
        "description": "Find files by name pattern, recursively from the workspace root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py' or 'test_*'"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time in ISO format.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "calculate",
        "description": "Evaluate a math expression safely. Supports +, -, *, /, **, %, and parentheses.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "A math expression, e.g. '2 + 3 * 4'"},
            },
            "required": ["expression"],
        },
    },
]


SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(node):
    """Evaluate a math AST without exec/eval."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPS:
        return SAFE_OPS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPS:
        return SAFE_OPS[type(node.op)](safe_eval(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


def resolve_safe_path(workspace: Path, rel_path: str) -> Path:
    """Resolve a path relative to workspace, rejecting any escape attempts."""
    target = (workspace / rel_path).resolve()
    if not str(target).startswith(str(workspace)):
        raise ValueError(f"path '{rel_path}' is outside the workspace")
    return target


def execute_tool(workspace: Path, name: str, tool_input: dict) -> tuple[str, bool]:
    """Run a tool and return (result_text, is_error)."""
    try:
        if name == "list_directory":
            target = resolve_safe_path(workspace, tool_input["path"])
            if not target.is_dir():
                return f"not a directory: {tool_input['path']}", True
            entries = []
            for item in sorted(target.iterdir()):
                kind = "DIR " if item.is_dir() else "FILE"
                size = "" if item.is_dir() else f" ({item.stat().st_size:,}B)"
                entries.append(f"  {kind} {item.name}{size}")
            if not entries:
                return f"directory '{tool_input['path']}' is empty", False
            return f"Contents of '{tool_input['path']}':\n" + "\n".join(entries), False

        if name == "read_file":
            target = resolve_safe_path(workspace, tool_input["path"])
            if not target.is_file():
                return f"not a file: {tool_input['path']}", True
            content = target.read_text(encoding="utf-8", errors="replace")
            if len(content) > 10_000:
                content = content[:10_000] + "\n\n[... truncated, file is larger than 10KB]"
            return content, False

        if name == "search_files":
            pattern = tool_input["pattern"]
            matches = [str(p.relative_to(workspace)) for p in workspace.rglob(pattern) if p.is_file()]
            if not matches:
                return f"no files matched pattern: {pattern}", False
            return f"Found {len(matches)} file(s):\n" + "\n".join(f"  {m}" for m in matches[:50]), False

        if name == "get_current_time":
            return datetime.now().isoformat(timespec="seconds"), False

        if name == "calculate":
            tree = ast.parse(tool_input["expression"], mode="eval")
            result = safe_eval(tree.body)
            return f"{tool_input['expression']} = {result}", False

        return f"unknown tool: {name}", True

    except Exception as e:
        return f"{type(e).__name__}: {e}", True


def run_agent_loop(client, messages: list[dict], workspace: Path, show_tools: bool) -> None:
    """Send messages, execute any tool calls, loop until Claude finishes."""
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        for block in response.content:
            if block.type == "text" and block.text:
                print(block.text, end="", flush=True)

        if response.stop_reason == "end_turn":
            print()
            messages.append({"role": "assistant", "content": response.content})
            return

        if response.stop_reason != "tool_use":
            print(f"\n[unexpected stop_reason: {response.stop_reason}]")
            messages.append({"role": "assistant", "content": response.content})
            return

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if show_tools:
                    print(f"\n  [calling {block.name}({block.input})]", flush=True)
                result_text, is_error = execute_tool(workspace, block.name, block.input)
                if show_tools:
                    preview = result_text[:100].replace("\n", " ")
                    print(f"  [→ {preview}{'...' if len(result_text) > 100 else ''}]", flush=True)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                    "is_error": is_error,
                })

        messages.append({"role": "user", "content": tool_results})


def main() -> None:
    parser = argparse.ArgumentParser(description="File system Q&A agent with tool use")
    parser.add_argument("--workspace", default=".", help="Root directory the agent can explore (default: cwd)")
    parser.add_argument("--show-tools", action="store_true", help="Print each tool call and result")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        print(f"Error: workspace not found: {workspace}", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()
    messages: list[dict] = []

    print(f"Agent ready (model: {MODEL})")
    print(f"Workspace: {workspace}")
    print(f"Tools: {', '.join(t['name'] for t in TOOLS)}")
    print("Commands: /quit, /reset, /tools (toggle tool trace)")
    print("-" * 60)

    show_tools = args.show_tools

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Goodbye!")
            break

        if user_input == "/reset":
            messages = []
            print("[Conversation reset]")
            continue

        if user_input == "/tools":
            show_tools = not show_tools
            print(f"[Tool trace: {'on' if show_tools else 'off'}]")
            continue

        messages.append({"role": "user", "content": user_input})
        print("\nAssistant: ", end="", flush=True)

        try:
            run_agent_loop(client, messages, workspace, show_tools)
        except anthropic.RateLimitError:
            print("\n[Rate limited. Please wait and try again.]")
            messages.pop()
        except anthropic.APIConnectionError:
            print("\n[Connection error.]")
            messages.pop()
        except anthropic.APIStatusError as e:
            print(f"\n[API error {e.status_code}: {e.message}]")
            messages.pop()


if __name__ == "__main__":
    main()
