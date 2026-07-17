"""Abstract base class for chat / LLM engines.

Chat engines handle query rewriting and response generation. They abstract
away whether the LLM is local (Ollama) or cloud (Gemini).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from deeplens.config import Settings


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class ChatResponse:
    """Response from the chat engine."""

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens


class ChatEngine(ABC):
    """Abstract chat / LLM engine.

    Implementations must support:
    - Single-turn generation (generate)
    - Multi-turn conversation (chat)
    - Streaming responses (stream)
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model / connection."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""
        ...

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str | None = None) -> ChatResponse:
        """Generate a single response from a prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system instruction.

        Returns:
            ChatResponse with the generated content.
        """
        ...

    @abstractmethod
    async def chat(
        self, messages: list[ChatMessage], system_prompt: str | None = None
    ) -> ChatResponse:
        """Multi-turn conversation.

        Args:
            messages: List of ChatMessage objects (conversation history).
            system_prompt: Optional system instruction.

        Returns:
            ChatResponse with the generated content.
        """
        ...

    @property
    def context_window(self) -> int:
        """Maximum input context size of the chat model, in tokens.

        Concrete engines override this to report their real limit (queried from
        model metadata at ``initialize()`` time). The default is a conservative
        8k so callers can size prompts safely without metadata.
        """
        return 8192

    async def rewrite_query(self, user_query: str, history: list[ChatMessage] | None = None) -> str:
        """Rewrite a user query into optimized search terms.

        Strips conversational filler, resolves pronouns from history,
        and emits dense semantic keywords.

        Args:
            user_query: The raw user query.
            history: Optional conversation history for context.

        Returns:
            Rewritten query optimized for vector search.
        """
        history_context = ""
        if history:
            recent = history[-4:]  # Last 2 turns
            history_context = "\n".join(
                f"{m.role}: {m.content}" for m in recent
            )
            history_context = f"\nRecent conversation:\n{history_context}\n"

        system = (
            "You are a query rewriting assistant. Your job is to transform a user's "
            "conversational query into optimized search keywords for a vector similarity search.\n\n"
            "Rules:\n"
            "1. Remove conversational filler (please, thank you, can you, etc.)\n"
            "2. Extract core semantic concepts and entities\n"
            "3. If conversation history is provided, resolve pronouns (it, that, etc.)\n"
            "4. Output ONLY the rewritten query — no explanation, no quotes\n"
            "5. Keep it concise but semantically rich\n"
        )

        prompt = f"{history_context}User query: {user_query}\n\nRewritten search query:"

        response = await self.generate(prompt, system_prompt=system)
        return response.content.strip().strip('"').strip("'")

    async def summarize_document(self, text: str, filename: str = "") -> str:
        """Produce a concise, content-level summary of a document.

        Used to build per-file ``summary`` vector records that enable
        document-level ("find the file that matches this description") retrieval.
        Implementations may override, but the default uses the generic
        ``generate`` endpoint with a summarization prompt.

        Args:
            text: Representative source text (already truncated by the caller).
            filename: Optional source filename for context.

        Returns:
            A concise summary (a few sentences).
        """
        system = (
            "You are a document summarization assistant. Given the beginning of a file, "
            "write a concise 2-4 sentence summary that captures what the document IS and "
            "its main subject matter (topic, genre, key entities, and intent).\n\n"
            "Rules:\n"
            "1. Focus on what the document is about, not generic boilerplate.\n"
            "2. Preserve distinctive proper nouns, names, places, and key themes.\n"
            "3. Output ONLY the summary — no preamble, no quotes, no headings.\n"
        )

        context = f"Filename: {filename}\n\nDocument excerpt:\n{text}\n\nSummary:"
        response = await self.generate(context, system_prompt=system)
        return response.content.strip().strip('"').strip("'")


def derive_summary_max_chars(
    chat_engine: "ChatEngine",
    *,
    default_chars: int = 4000,
    min_chars: int = 500,
    max_chars: int = 20000,
    chars_per_token: int = 4,
    system_overhead: int = 300,
    summary_output_reserve: int = 512,
) -> int:
    """Compute a safe character budget for summarization from the model context.

    Leaves room for the system prompt and the generated summary so the excerpt
    handed to ``summarize_document`` never exceeds the chat model's input limit.

    Args:
        chat_engine: The (initialized) chat engine; its ``context_window`` drives the budget.
        default_chars: Fallback when the context window is unknown.
        min_chars / max_chars: Clamp the result to a sane range.
        chars_per_token: Approximate characters per token (English); conservative.
        system_overhead: Tokens reserved for the system prompt + boilerplate.
        summary_output_reserve: Tokens reserved for the model's summary output.

    Returns:
        The character budget to use for ``settings.summary_max_chars``.
    """
    ctx = chat_engine.context_window
    if not ctx or ctx <= 0:
        return default_chars
    budget_tokens = max(0, ctx - system_overhead - summary_output_reserve)
    chars = int(budget_tokens * chars_per_token)
    return max(min_chars, min(chars, max_chars))


def configure_summary_budget(settings: Settings, chat_engine: "ChatEngine") -> int:
    """Set ``settings.summary_max_chars`` from the chat model's context window.

    Call once after ``chat_engine.initialize()``. If the context window is
    unknown, the existing default is preserved. Returns the chosen value.
    """
    value = derive_summary_max_chars(chat_engine)
    settings.summary_max_chars = value
    return value
