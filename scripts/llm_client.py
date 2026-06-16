import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from scripts.logger import get_logger

logger = get_logger("llm_client")

DEFAULT_TEMPERATURE = 0.0

# OpenAI-compatible LLM gateway. Configure via env (see config/.env.example):
#   LLM_BASE_URL, LLM_MODEL, LLM_API_KEY (OPENAI_API_KEY also accepted).
# Load config/.env if python-dotenv is available (no-op if absent or file missing).
try:
    from dotenv import load_dotenv

    load_dotenv(Path("config") / ".env")
except Exception:
    pass


def _api_key() -> str:
    key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "LLM API key not set. Add LLM_API_KEY to config/.env "
            "(see config/.env.example)."
        )
    return key


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} not set. Add it to config/.env (see config/.env.example).")
    return value


def get_llm(model: str | None = None, temperature: float = DEFAULT_TEMPERATURE, **kwargs) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or _required("LLM_MODEL"),
        base_url=_required("LLM_BASE_URL"),
        api_key=_api_key(),
        temperature=temperature,
        **kwargs,
    )


def llm_call(prompt: str, system: str | None = None, model: str | None = None,
             temperature: float = DEFAULT_TEMPERATURE, **kwargs) -> str:
    llm = get_llm(model=model, temperature=temperature, **kwargs)
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    resp = llm.invoke(messages)
    return resp.content


def llm_json(prompt: str, system: str | None = None, model: str | None = None,
             temperature: float = DEFAULT_TEMPERATURE, **kwargs) -> dict:
    llm = get_llm(model=model, temperature=temperature, **kwargs)
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    chain = llm | JsonOutputParser()
    return chain.invoke(messages)
