import os
import sys

import anthropic

MODEL = "claude-opus-4-7"
SYSTEM_PROMPT = "You are a helpful, friendly assistant. Respond concisely in the user's language."


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it with: export ANTHROPIC_API_KEY='your-api-key'", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()
    messages: list[dict] = []

    print(f"Chatbot ready (model: {MODEL})")
    print("Type your message. Commands: /quit, /reset, /history")
    print("-" * 50)

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
            print("[Conversation cleared]")
            continue

        if user_input == "/history":
            print(f"[{len(messages)} messages in history]")
            for i, msg in enumerate(messages, 1):
                content = msg["content"] if isinstance(msg["content"], str) else "<complex>"
                preview = content[:60] + "..." if len(content) > 60 else content
                print(f"  {i}. {msg['role']}: {preview}")
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

            if final.stop_reason == "max_tokens":
                print("[Note: response was truncated due to max_tokens limit]")

        except anthropic.RateLimitError:
            print("\n[Rate limited. Please wait a moment and try again.]")
            messages.pop()
        except anthropic.APIConnectionError:
            print("\n[Connection error. Check your internet connection.]")
            messages.pop()
        except anthropic.APIStatusError as e:
            print(f"\n[API error {e.status_code}: {e.message}]")
            messages.pop()


if __name__ == "__main__":
    main()
