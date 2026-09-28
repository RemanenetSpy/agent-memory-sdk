# Examples

See `examples/basic_usage.py` and evaluation datasets in `benchmarks/datasets/`.

## Support Bot with Memory Decisions

See `examples/support_bot.py` for a support-bot example that uses memory decisions to determine how to answer a user query.

The example handles four possible decisions:

- **Replay** — returns the stored response directly.
- **Restore** — restores relevant context before generating a response.
- **Verify** — uses a verification tool before generating a response.
- **None** — makes a fresh LLM call when no useful memory is available.

The LLM and verification tool are stubbed in the example so the focus is on how the memory decisions are handled.
