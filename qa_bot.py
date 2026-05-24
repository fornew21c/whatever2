import argparse
import base64
import os
import sys
from pathlib import Path

import anthropic

MODEL = "claude-opus-4-7"
SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer questions based on the provided documents. "
    "When citing information, mention which document it came from. "
    "If the answer isn't in the documents, say so clearly instead of guessing. "
    "Respond concisely in the user's language."
)

TEXT_EXTENSIONS = {".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".log", ".html", ".xml", ".yaml", ".yml"}


def load_document(path: Path) -> dict:
    """Load a file into an Anthropic content block."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": data},
            "title": path.name,
            "cache_control": {"type": "ephemeral"},
        }

    if suffix in TEXT_EXTENSIONS or suffix == "":
        text = path.read_text(encoding="utf-8", errors="replace")
        return {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": text},
            "title": path.name,
            "cache_control": {"type": "ephemeral"},
        }

    raise ValueError(f"Unsupported file type: {suffix} (path: {path})")


def build_initial_messages(doc_blocks: list[dict]) -> list[dict]:
    """Build the seed conversation containing the documents."""
    intro_text = f"I'm sharing {len(doc_blocks)} document(s) with you. Please answer my questions based on them."
    return [
        {"role": "user", "content": [*doc_blocks, {"type": "text", "text": intro_text}]},
        {"role": "assistant", "content": "I've reviewed the documents. What would you like to know?"},
    ]


def format_usage(usage) -> str:
    """Format cache stats from a response."""
    return (
        f"in={usage.input_tokens} "
        f"cache_write={usage.cache_creation_input_tokens} "
        f"cache_read={usage.cache_read_input_tokens} "
        f"out={usage.output_tokens}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Document Q&A chatbot with prompt caching")
    parser.add_argument("docs", nargs="+", help="Path(s) to documents (PDF or text)")
    parser.add_argument("--show-usage", action="store_true", help="Show token usage after each response")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    doc_blocks = []
    for doc_path_str in args.docs:
        doc_path = Path(doc_path_str)
        if not doc_path.is_file():
            print(f"Error: file not found: {doc_path}", file=sys.stderr)
            sys.exit(1)
        try:
            doc_blocks.append(load_document(doc_path))
            print(f"Loaded: {doc_path.name} ({doc_path.stat().st_size:,} bytes)")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    client = anthropic.Anthropic()
    initial_messages = build_initial_messages(doc_blocks)
    messages = list(initial_messages)

    print(f"\nQ&A bot ready (model: {MODEL})")
    print("Documents are cached after the first request — subsequent queries are ~90% cheaper.")
    print("Commands: /quit, /reset, /history, /usage")
    print("-" * 60)

    last_usage = None

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
            messages = list(initial_messages)
            print("[Conversation reset (documents still loaded)]")
            continue

        if user_input == "/history":
            qa_count = (len(messages) - len(initial_messages)) // 2
            print(f"[{qa_count} Q&A turn(s) since reset; documents loaded: {len(doc_blocks)}]")
            continue

        if user_input == "/usage":
            if last_usage is None:
                print("[No requests sent yet]")
            else:
                print(f"[Last request: {format_usage(last_usage)}]")
            continue

        messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        assistant_response = ""

        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    assistant_response += text

                final = stream.get_final_message()

            print()
            messages.append({"role": "assistant", "content": assistant_response})
            last_usage = final.usage

            if args.show_usage:
                print(f"[tokens: {format_usage(final.usage)}]")

            if final.stop_reason == "max_tokens":
                print("[Note: response was truncated]")

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
