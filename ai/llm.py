"""Shared LLM configuration for the ai/ scripts.

Everything here routes through OpenRouter, which speaks the OpenAI wire format,
so we keep the langchain-openai classes and just point them at a different
base_url. To change provider or model, edit .env - no code changes needed.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# OpenRouter model IDs are namespaced by the upstream provider.
CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")


def get_api_key():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to ai/.env - "
            "create a key at https://openrouter.ai/keys"
        )
    return api_key


def get_chat_model(temperature=0.0, model=None, **kwargs):
    """A chat model pointed at OpenRouter.

    Note: the scripts call .with_structured_output(...) on the result, which
    defaults to method="json_schema" and needs a model that supports strict
    structured outputs. gpt-4o-mini does. If you swap CHAT_MODEL for one that
    doesn't, pass method="function_calling" at the call site instead.
    """
    return ChatOpenAI(
        model=model or CHAT_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=get_api_key(),
        temperature=temperature,
        **kwargs,
    )


def get_embeddings():
    """An embeddings client pointed at OpenRouter."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OPENROUTER_BASE_URL,
        api_key=get_api_key(),
        # OpenRouter expects raw text. The default (True) makes LangChain send
        # tiktoken token IDs instead, which only lines up with the provider's
        # tokenizer for openai/* models - anything else gets silently embedded
        # as nonsense.
        check_embedding_ctx_length=False,
    )
