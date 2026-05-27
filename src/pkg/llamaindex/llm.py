"""LocalHFLLM — local HuggingFace LLM for relay's LlamaIndex RAG layer."""

from collections.abc import AsyncGenerator, Generator, Sequence
from typing import Any, Callable, Optional

from pydantic import PrivateAttr

import torch
from llama_index.core.base.llms.base import LLMMetadata
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    MessageRole,
)
from llama_index.core.llms import LLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from transformers import AutoModelForCausalLM, AutoTokenizer

from relay.config import CONFIG

DEFAULT_LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_HF_ROLE_MAP = {
    MessageRole.USER: "user",
    MessageRole.ASSISTANT: "assistant",
    MessageRole.SYSTEM: "system",
}


class LocalHFLLM(LLM):
    """LlamaIndex LLM backed by a local HuggingFace transformers model.

    Args:
        model_id: HuggingFace model name.  Defaults to
            ``TinyLlama/TinyLlama-1.1B-Chat-v1.0``.
    """

    model_id: str = DEFAULT_LLM_MODEL

    @classmethod
    def class_name(cls) -> str:
        return "LocalHFLLM"

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()

    def __init__(self, model_id: str = DEFAULT_LLM_MODEL) -> None:
        super().__init__()
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(
            self,
            "_tokenizer",
            AutoTokenizer.from_pretrained(model_id, trust_remote_code=True),
        )
        object.__setattr__(
            self,
            "_model",
            AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True),
        )

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model_id, is_chat_model=True)

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        hf_messages = [
            {"role": _HF_ROLE_MAP.get(m.role, "user"), "content": m.content or ""}
            for m in messages
        ]
        prompt = self._tokenizer.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=192, do_sample=False
            )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = text[len(prompt) :].strip()
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=reply),
        )

    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        return self.chat(list(messages), **kwargs)

    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError("stream_chat not implemented in LocalHFLLM")

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> AsyncGenerator[ChatResponse, None]:
        raise NotImplementedError("astream_chat not implemented in LocalHFLLM")

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs, max_new_tokens=192, do_sample=False
            )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = text[len(prompt) :].strip()
        return CompletionResponse(text=reply)

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self.complete(prompt, formatted=formatted, **kwargs)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Generator[CompletionResponse, None, None]:
        raise NotImplementedError("stream_complete not implemented in LocalHFLLM")

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> AsyncGenerator[CompletionResponse, None]:
        raise NotImplementedError("astream_complete not implemented in LocalHFLLM")


def get_llm(
    provider: Optional[str] = None,
    model_id: str = DEFAULT_LLM_MODEL,
) -> LLM:
    """Build an LLM for the given provider.

    Args:
        provider: One of ``"local"``, ``"openai"``, ``"anthropic"``.
            Defaults to ``RELAY_LLM_PROVIDER`` env var (or ``"local"``).
        model_id: Model identifier to pass to the provider.

    Returns:
        A LlamaIndex ``LLM`` instance.

    Raises:
        RuntimeError: If a cloud provider is specified but its extra package
            is not installed (e.g. ``llama-index-llms-openai``).
    """
    resolved = (provider or CONFIG.llm_provider).lower()

    if resolved == "local":
        return LocalHFLLM(model_id=model_id)

    if resolved == "openai":
        try:
            from llama_index.llms.openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "OpenAI provider requires `llama-index-llms-openai`. "
                "Install: `uv sync --extra openai`"
            )
        return OpenAI(
            model=CONFIG.openai_model if model_id == DEFAULT_LLM_MODEL else model_id,
            api_key=CONFIG.openai_api_key or None,
        )

    if resolved == "anthropic":
        try:
            from llama_index.llms.anthropic import Anthropic
        except ImportError:
            raise RuntimeError(
                "Anthropic provider requires `llama-index-llms-anthropic`. "
                "Install: `uv sync --extra anthropic`"
            )
        return Anthropic(
            model=CONFIG.anthropic_model if model_id == DEFAULT_LLM_MODEL else model_id,
            api_key=CONFIG.anthropic_api_key or None,
        )

    raise ValueError(
        f"Unknown LLM provider: {resolved!r}. Use local, openai, or anthropic."
    )


def create_query_engine(
    tenant_id: str = CONFIG.default_tenant,
    at: Optional[str] = None,
    epoch_id: Optional[int] = None,
    top_k: int = CONFIG.default_top_k,
    retrieval_policy: str = CONFIG.default_retrieval_policy,
    text_resolver: Optional[Callable[[str, Optional[str]], str]] = None,
    llm_provider: Optional[str] = None,
    llm_model_id: str = DEFAULT_LLM_MODEL,
) -> RetrieverQueryEngine:
    """Build a ready-to-use ``RetrieverQueryEngine`` backed by relay.

    Args:
        tenant_id: Tenant namespace for isolation.
        at: ISO timestamp for time-travel queries.
        epoch_id: Pin to a specific epoch (overrides ``at``).
        top_k: Number of results to return.
        retrieval_policy: ``"dense"`` or ``"hybrid"``.
        text_resolver: Callback ``(doc_id, source_file) -> str`` that loads
            document text from your store.  Required for LLM synthesis to work.
        llm_provider: ``"local"`` (HF), ``"openai"``, or ``"anthropic"``.
            Defaults to ``RELAY_LLM_PROVIDER`` env var.
        llm_model_id: Model identifier for the provider.

    Returns:
        A ``RetrieverQueryEngine`` instance.  Call ``.query("...")`` on it.

    Raises:
        RuntimeError: If the LLM provider fails to load.
    """
    from pkg.llamaindex.retriever import RelayRetriever

    retriever = RelayRetriever(
        tenant_id=tenant_id,
        at=at,
        epoch_id=epoch_id,
        top_k=top_k,
        retrieval_policy=retrieval_policy,
        text_resolver=text_resolver,
    )
    llm = get_llm(provider=llm_provider, model_id=llm_model_id)
    synthesizer = CompactAndRefine(llm=llm, streaming=False)
    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
    )
