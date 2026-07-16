"""Abstract base class for chat / LLM engines.

Chat engines handle query rewriting and response generation. They abstract
away whether the LLM is local (Ollama) or cloud (Gemini).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
