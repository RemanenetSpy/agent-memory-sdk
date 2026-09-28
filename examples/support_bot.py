"""Support bot example using memory decisions."""

from agent_memory import Memory, MemoryAction, MemoryType


def call_llm(prompt: str) -> str:
    """Stub for an LLM call."""
    return f"LLM response: {prompt}"


def verify_with_tool(context: str) -> str:
    """Stub for an external verification tool."""
    return f"Verified information: {context}"


def support_bot(memory: Memory, query: str) -> str:
    decision = memory.resolve(query)

    if decision.action == MemoryAction.REPLAY:
        return decision.response

    if decision.action == MemoryAction.RESTORE:
        context = memory.format_restore_context(decision)
        return call_llm(
            f"Use the following restored context to answer the user:\n"
            f"{context}\n\nUser: {query}"
        )

    if decision.action == MemoryAction.VERIFY:
        context = memory.format_verify_context(decision)
        verified = verify_with_tool(context)
        return call_llm(
            f"Use this verified information to answer the user:\n"
            f"{verified}\n\nUser: {query}"
        )

    return call_llm(query)


def main() -> None:
    memory = Memory(persist_dir=".support_bot_demo")

    memory.remember(
        query="How do I reset my password?",
        response="Go to Settings → Security → Reset Password and follow the email link.",
        type=MemoryType.CONVERSATION,
        tags=["auth", "faq"],
    )

    memory.remember(
        query="Current API rate limit",
        response="1000 requests/minute per API key.",
        type=MemoryType.FACT,
        requires_verification=True,
    )

    queries = [
        "How do I reset my password?",
        "Give me a one-sentence explanation of password reset",
        "What is the API rate limit?",
        "What's the weather today?",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        print(f"Bot: {support_bot(memory, query)}")


if __name__ == "__main__":
    main()


