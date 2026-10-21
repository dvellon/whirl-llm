# Benchmark prompts (v0.2.0)

The prompt files behind the decode numbers in [docs/benchmarks.md](../../docs/benchmarks.md#prompts).

For the reproducible Radeon 8060S 8K/1K performance run, see
[`gfx1151_public_8k.ps1`](gfx1151_public_8k.ps1) and the
[`gfx1151` performance lab](../../docs/gfx1151-performance-lab.md). That harness
records its exact source commit, hardware inventory, timings, logs, and output
digests; its `--fast` result uses relaxed MTP acceptance and is reported as such.

| File | Used for | Content |
|---|---|---|
| `prompts/coding_chat_zh.txt` | "coding chat (new text)" rows: 512 generated tokens, one 64-token warm-up request first | one Chinese coding question (write `dedup.py` with tests); the answer is all newly written, the prompt holds no code to copy. Written by us (Apache-2.0, as this repository) |
| `prompts/document_qa_question.txt` | "document QA" rows (2k to ~250k tokens, 256 generated tokens) | the question appended after the long document, identical at every length. Written by us |
| `prompts/agent_fixture.json` | coding-agent session (§6 of benchmarks.md) | `system`: the coding-agent system prompt (instructions plus WHIRL documentation pages used as "skills / MCP tool manuals"; WHIRL documentation as of 2026-10-16, Apache-2.0). `files.<name>.orig`: the first 12,000 characters of four llama.cpp source files (`src/llama-batch.cpp`, `llama-chat.cpp`, `llama-mmap.cpp`, `llama-adapter.cpp`; MIT License, Copyright (c) 2023-2026 The ggml authors), returned by the `read_file` tool. `files.<name>.mod`: the same text with one added comment line, sent by the user in turn 4 |

The long documents of the document-QA prompts are not redistributed here, because they are made of third-party text under their own licenses; their composition is described in benchmarks.md §2.1.

## Agent session protocol

Two sessions run one after another on one server (`-np 1`, temperature 0, seed 1, thinking off, at most 384 tokens per call, full history resent on every call), after at least 25 s of warm-up requests (unique 2k-token prompts) before session 1. Session 1 uses `llama-batch.cpp`, session 2 `llama-chat.cpp`; both share the same system prompt, so session 2 can reuse the cached prefix. Tool definition: one function `read_file(path)`. Turns:

1. `Read src/<file> and tell me in two sentences what it is responsible for.` The model calls `read_file`; the tool result is `files.<file>.orig`; the model answers.
2. `Which function in that file looks the most complex, and why? Two sentences.`
3. `Rewrite that function with one concrete refactoring applied (for example extracting a helper). Output only the C++ code.`
4. `I applied an edit to src/<file>. Here is the updated file:` followed by `files.<file>.mod` in a `cpp` code block, then `Review the change: is anything broken or inconsistent? Answer in three sentences.`

Whenever the model answers with a `read_file` call instead of text, the tool result is sent and the model is called again (at most 3 tool calls in a row) (this is why engines can make different numbers of calls). Wall time of a session = from the first request to the last response.
