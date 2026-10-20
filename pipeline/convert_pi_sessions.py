"""Turn mu/pi session files (~/.mu/agent/sessions/**/*.jsonl) into transcripts mu's exporter replays.

A session file holds one entry per line; entries of type "message" carry the conversation:
  user        content: str | [{type: "text", text}...]
  assistant   content: [{type: "text"} | {type: "thinking"} | {type: "toolCall", id, name, arguments}]
  toolResult  toolCallId, toolName, content, isError
The transcript keeps mu's own tool names (bash, read, edit, write, ...) and each result's error flag,
and follows the file in order (branches a session forked into are flattened in file order).

  uv run pipeline/convert_pi_sessions.py ~/.mu/agent/sessions data/transcripts/mu-sessions.jsonl
"""

import argparse
import json
import os


def text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")


def convert(path):
    messages = []
    with open(path, encoding="utf-8", errors="replace") as source:
        for line in source:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "message":
                continue
            message = entry.get("message") or {}
            role = message.get("role")
            if role == "user":
                text = text_of(message.get("content"))
                if text.strip():
                    messages.append({"role": "user", "content": text})
            elif role == "assistant":
                content = message.get("content") or []
                calls = [
                    {"id": block.get("id", ""), "name": block.get("name", ""), "arguments": block.get("arguments") or {}}
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "toolCall"
                ]
                out = {"role": "assistant", "content": text_of(content), "stop_reason": message.get("stopReason")}
                if calls:
                    out["tool_calls"] = calls
                messages.append(out)
            elif role == "toolResult":
                messages.append(
                    {
                        "role": "tool",
                        "content": text_of(message.get("content")),
                        "tool_call_id": message.get("toolCallId", ""),
                        "name": message.get("toolName", ""),
                        "is_error": bool(message.get("isError")),
                    }
                )
    return messages


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", help="directory of session files")
    parser.add_argument("out")
    parser.add_argument("--min-user-messages", type=int, default=1)
    args = parser.parse_args()

    kept = skipped = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for root, _, files in os.walk(args.sessions):
            for name in sorted(files):
                if not name.endswith(".jsonl"):
                    continue
                path = os.path.join(root, name)
                messages = convert(path)
                users = sum(1 for message in messages if message["role"] == "user")
                if users < args.min_user_messages:
                    skipped += 1
                    continue
                record = {
                    "id": os.path.relpath(path, args.sessions),
                    "source": "mu-sessions",
                    "repo": os.path.basename(root),
                    "messages": messages,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
    print(f"kept {kept} sessions, skipped {skipped} without user messages")


if __name__ == "__main__":
    main()
