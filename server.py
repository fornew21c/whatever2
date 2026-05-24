import json
import os
import sys
from pathlib import Path

import anthropic
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

MODEL = "claude-opus-4-7"
SYSTEM_PROMPT = "You are a helpful, friendly assistant. Respond concisely in the user's language."

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)

app = FastAPI(title="Claude Chat")
client = anthropic.Anthropic()


def sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    if not messages or messages[-1].get("role") != "user":
        return {"error": "messages must end with a user turn"}

    def stream():
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as s:
                for text in s.text_stream:
                    yield sse_event({"text": text})
                final = s.get_final_message()
            yield sse_event({"done": True, "stop_reason": final.stop_reason})
        except anthropic.RateLimitError:
            yield sse_event({"error": "Rate limited. Please wait and try again."})
        except anthropic.APIStatusError as e:
            yield sse_event({"error": f"API error {e.status_code}: {e.message}"})
        except Exception as e:
            yield sse_event({"error": f"{type(e).__name__}: {e}"})

    return StreamingResponse(stream(), media_type="text/event-stream")


static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
