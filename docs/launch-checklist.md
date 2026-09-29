# Launch checklist

Everything below is prepared in this repo; each step needs your account and
is one command or one click. Order matters roughly top to bottom.

## 0. Commit and push the pending work

In the working tree: the LongMemEval benchmark harness + report + charts
(`benchmarks/longmemeval/`), the decision-layer improvements
(`policy.py` / `retriever.py`: replay margin, session-diversified retrieval),
and README/docs updates with the measured numbers. Review, commit, push.
The blog post is in the website repo:
`theprodsde.github.io/content/blog/agent-memory-decision-layer.mdx`
(its numbers section was updated with the benchmark results — re-read before
publishing).

## 1. Enable GitHub Discussions — ✅ DONE (2026-09-21)

Discussions are enabled and the roadmap thread is live:
https://github.com/theprodsde/agent-memory-sdk/discussions/1

Remaining manual click: open that discussion and choose **Pin discussion**
in the right sidebar (GitHub removed the API for pinning discussions).

Starter issues are also created and labeled (#2 trap cases, #3 LangChain
adapter, #4 LongMemEval harness, #5 worked example).

## 2. Publish to the MCP Registry

`server.json` at the repo root is already validated against the official
2025-09-29 schema, and the `agent-memory-sdk` console script exists so the
registry's default `uvx agent-memory-sdk` launch works.

Prerequisite: the version in `server.json` must exist on PyPI, and the
registry verifies PyPI ownership via a README marker — make sure the README
on the published PyPI version contains the string:
`mcp-name: io.github.theprodsde/agent-memory`
(add it near the bottom of README.md, cut a release, then publish).

```bash
brew install mcp-publisher
mcp-publisher login github        # authenticates io.github.theprodsde/* namespace
mcp-publisher publish             # reads ./server.json
```

Docs: https://github.com/modelcontextprotocol/registry/blob/main/docs/guides/publishing/publish-server.md

## 3. Publish the blog post

The post is written in your site's format at
`content/blog/agent-memory-decision-layer.mdx`. Push the site repo and it
goes live at `/blog/agent-memory-decision-layer`.

## 4. Show HN / r/LocalLLaMA

Suggested Show HN title (under 80 chars):

> Show HN: Agent Memory – local memory SDK with published retrieval results

Alternative (the honesty angle, often stronger on HN):

> Show HN: We benchmarked our memory SDK and published the parts that failed

Body structure (in this order):

1. One line on what it is: a memory layer whose `resolve()` returns an
   explicit REPLAY / RESTORE / VERIFY / NONE decision instead of silently
   injecting context.
2. The measured card: **87.0% session Recall@5 on cleaned LongMemEval_M** with
   turn-pair indexing, zero LLM calls, and 12.05ms p50 against ~2,500-entry
   haystacks. State that this is a retrieval proxy, not a paper-baseline or
   end-to-end comparison. Link the
   [benchmark report](../benchmarks/longmemeval/REPORT.md).
3. The honesty card: what we found and published anyway — retrieval-level
   abstention failed calibration (gate ships off), semantic thresholds were
   miscalibrated, and our own flagship trap eval is 34/36, not 100%. HN
   rewards this more than any headline number.
4. Reproduce the semantic `_S` result: `uv run python benchmarks/longmemeval/run_retrieval.py --semantic`.
5. Invite: submit trap cases that break the decision layer; the
   repeated-query decision-layer benchmark is open for contributions.

Do NOT claim: end-to-end accuracy superiority over Mem0/Zep (not measured),
or "fixes context rot" (README wording is "prevents context pollution").

Post the same piece to r/LocalLLaMA with flair "Resources" — lead with the
zero-API-keys/local-ONNX angle there. Best posting windows: Tue–Thu,
14:00–16:00 UTC.

## 5. After launch

- Respond to the first issues/comments within hours — early responsiveness
  is the strongest liveness signal.
- Label 3–4 issues `good first issue` from CONTRIBUTING.md's list (eval trap
  cases, LangChain adapter, worked examples). Close/update issue #4
  (LongMemEval harness) — it shipped; replace it with the follow-ups from the
  report roadmap: the LLM-judge end-to-end stage, the semantic `_M` run, and
  per-mode threshold calibration.
- Add the GIF and registry badge to the PyPI page by cutting a release with
  the updated README.
