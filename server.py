import json
import os
import sys
from pathlib import Path

import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent import TOOLS, execute_tool

MODEL = "claude-opus-4-7"
SYSTEM_PROMPT = (
    "You are a helpful assistant with read-only tools to explore a workspace "
    "(list_directory, read_file, search_files, get_current_time, calculate). "
    "Use the tools when they help answer the user's question. Respond concisely "
    "in the user's language."
)
MAX_AGENT_ITERATIONS = 10

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)

WORKSPACE = Path(os.environ.get("WORKSPACE", ".")).resolve()
if not WORKSPACE.is_dir():
    print(f"Error: workspace not found: {WORKSPACE}", file=sys.stderr)
    sys.exit(1)

app = FastAPI(title="Claude Chat")
client = anthropic.Anthropic()


def sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_one_turn(messages: list[dict]):
    """Stream a single API call, yielding SSE events for text and tool_use blocks."""
    blocks_in_progress: dict[int, dict] = {}

    with client.messages.stream(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    ) as stream:
        for event in stream:
            if event.type == "content_block_start":
                block = event.content_block
                if block.type == "tool_use":
                    blocks_in_progress[event.index] = {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input_json": "",
                    }
            elif event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "text_delta":
                    yield sse_event({"type": "text", "text": delta.text})
                elif delta.type == "input_json_delta":
                    if event.index in blocks_in_progress:
                        blocks_in_progress[event.index]["input_json"] += delta.partial_json
            elif event.type == "content_block_stop":
                block = blocks_in_progress.pop(event.index, None)
                if block and block["type"] == "tool_use":
                    try:
                        parsed = json.loads(block["input_json"]) if block["input_json"] else {}
                    except json.JSONDecodeError:
                        parsed = {"_raw": block["input_json"]}
                    yield sse_event({
                        "type": "tool_use",
                        "id": block["id"],
                        "name": block["name"],
                        "input": parsed,
                    })

        final = stream.get_final_message()

    return final


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = list(body.get("messages", []))

    if not messages or messages[-1].get("role") != "user":
        return {"error": "messages must end with a user turn"}

    def stream():
        try:
            for iteration in range(MAX_AGENT_ITERATIONS):
                gen = stream_one_turn(messages)
                final = None
                try:
                    while True:
                        yield next(gen)
                except StopIteration as stop:
                    final = stop.value

                serialized_content = [block.model_dump() for block in final.content]
                messages.append({"role": "assistant", "content": serialized_content})

                if final.stop_reason == "end_turn":
                    yield sse_event({"type": "done", "stop_reason": "end_turn"})
                    return

                if final.stop_reason != "tool_use":
                    yield sse_event({"type": "done", "stop_reason": final.stop_reason})
                    return

                tool_results = []
                for block in final.content:
                    if block.type == "tool_use":
                        result_text, is_error = execute_tool(WORKSPACE, block.name, block.input)
                        yield sse_event({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                            "is_error": is_error,
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                            "is_error": is_error,
                        })

                messages.append({"role": "user", "content": tool_results})

            yield sse_event({"type": "error", "error": f"Agent loop exceeded {MAX_AGENT_ITERATIONS} iterations"})

        except anthropic.RateLimitError:
            yield sse_event({"type": "error", "error": "Rate limited. Please wait and try again."})
        except anthropic.APIStatusError as e:
            yield sse_event({"type": "error", "error": f"API error {e.status_code}: {e.message}"})
        except Exception as e:
            yield sse_event({"type": "error", "error": f"{type(e).__name__}: {e}"})

    return StreamingResponse(stream(), media_type="text/event-stream")


static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
