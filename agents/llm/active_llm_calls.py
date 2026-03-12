"""
Centralized active-LLM call layer.

Routes all app LLM calls through get_active_llm() and implements each provider
(Claude, GPT-4, Gemini, Ollama, HuggingFace) with clear try/except and raise on failure.
Used by agents/llm/llm_interface.py generate_response() and trading/services/chat_nl_service.py call_claude().
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default test prompt for Admin "Test Connection"
TEST_PROMPT = "Reply with exactly: OK"


def get_provider_status(provider: str, *, huggingface_mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Return status for a given provider: configured (bool), message (str).
    For Ollama, checks if http://localhost:11434 is reachable.
    For HuggingFace, can report inference (API key) and/or local (transformers) status.
    """
    from config.llm_config import get_llm_config, LLM_PROVIDERS
    if provider not in LLM_PROVIDERS:
        return {"configured": False, "message": f"Unknown provider: {provider}"}
    cfg = get_llm_config()
    if provider == "claude":
        return {"configured": bool(cfg.anthropic_api_key), "message": "API key set" if cfg.anthropic_api_key else "ANTHROPIC_API_KEY not set"}
    if provider == "gpt4":
        return {"configured": bool(cfg.openai_api_key), "message": "API key set" if cfg.openai_api_key else "OPENAI_API_KEY not set"}
    if provider == "gemini":
        return {"configured": bool(cfg.google_api_key), "message": "API key set" if cfg.google_api_key else "GOOGLE_API_KEY / GEMINI_API_KEY not set"}
    if provider == "kimi":
        return {"configured": bool(cfg.moonshot_api_key), "message": "API key set" if cfg.moonshot_api_key else "MOONSHOT_API_KEY not set"}
    if provider == "huggingface":
        inference_ok = bool(cfg.huggingface_api_key)
        try:
            import transformers  # noqa: F401
            local_ok = True
        except ImportError:
            local_ok = False
        mode = huggingface_mode or "inference"
        if mode == "local":
            configured = local_ok
            message = "transformers available" if local_ok else "transformers not installed (pip install transformers)"
        else:
            configured = inference_ok
            message = "API key set (free tier rate-limited)" if inference_ok else "HUGGINGFACE_API_KEY / HF_TOKEN not set"
        return {"configured": configured, "message": message, "inference_ok": inference_ok, "local_ok": local_ok}
    if provider == "ollama":
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as _:
                pass
            return {"configured": True, "message": "Ollama running"}
        except Exception as e:
            return {"configured": False, "message": "Ollama not running (start with: ollama serve)"}
    return {"configured": False, "message": "Unknown provider"}


def test_active_llm() -> tuple:
    """
    Send a simple test prompt to the active LLM. Returns (success: bool, message: str).
    Used by Admin "Test Connection" button.
    """
    try:
        out = call_active_llm_simple(TEST_PROMPT, max_tokens=50)
        return True, out or "OK"
    except Exception as e:
        return False, str(e)


def get_active_llm():
    from config.llm_config import get_active_llm as _get
    return _get()


def _is_anthropic_401(e: Exception) -> bool:
    """True if the exception indicates Anthropic 401 Unauthorized."""
    code = getattr(e, "status_code", None)
    if code == 401:
        return True
    msg = (getattr(e, "message", None) or str(e)).lower()
    return "401" in msg or "unauthorized" in msg


def _call_claude_simple(prompt: str, model: str, *, max_tokens: int = 2048) -> str:
    from config.llm_config import get_anthropic_client
    try:
        client = get_anthropic_client()
    except Exception as e:
        logger.exception("Claude client init failed: %s", e)
        raise
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        return (msg.content[0].text if msg.content else "").strip()
    except Exception as e:
        if _is_anthropic_401(e):
            logger.warning("Anthropic key invalid, falling back to OpenAI")
            return _call_gpt4_simple(prompt, "gpt-4o", max_tokens=max_tokens)
        logger.exception("Claude API call failed: %s", e)
        raise


def _call_claude_chat(
    system_prompt: str,
    context_block: str,
    conversation_messages: List[Dict[str, str]],
    user_message: str,
    model: str,
    *,
    max_tokens: int = 2048,
) -> str:
    from config.llm_config import get_anthropic_client
    try:
        client = get_anthropic_client()
    except Exception as e:
        logger.exception("Claude client init failed: %s", e)
        raise
    full_system = f"{system_prompt}\n\n{context_block}"
    api_messages: List[Dict[str, Any]] = []
    for m in conversation_messages[-12:]:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        api_messages.append({"role": "user" if role == "user" else "assistant", "content": content})
    api_messages.append({"role": "user", "content": user_message})
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,
            system=full_system,
            messages=api_messages,
        )
        text = resp.content[0].text if resp.content else ""
        return text.strip() or ""
    except Exception as e:
        if _is_anthropic_401(e):
            logger.warning("Anthropic key invalid, falling back to OpenAI")
            return _call_gpt4_chat(system_prompt, context_block, conversation_messages, user_message, "gpt-4o", max_tokens=max_tokens)
        logger.exception("Claude API chat failed: %s", e)
        raise


def _call_gpt4_simple(prompt: str, model: str, *, max_tokens: int = 2048) -> str:
    from config.llm_config import get_openai_client
    try:
        client = get_openai_client()
    except Exception as e:
        logger.exception("OpenAI client init failed: %s", e)
        raise
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp.choices[0].message.content if resp.choices else "") or ""
        return text.strip()
    except Exception as e:
        logger.exception("OpenAI API call failed: %s", e)
        raise


def _call_gpt4_chat(
    system_prompt: str,
    context_block: str,
    conversation_messages: List[Dict[str, str]],
    user_message: str,
    model: str,
    *,
    max_tokens: int = 2048,
) -> str:
    from config.llm_config import get_openai_client
    try:
        client = get_openai_client()
    except Exception as e:
        logger.exception("OpenAI client init failed: %s", e)
        raise
    full_system = f"{system_prompt}\n\n{context_block}"
    messages = [{"role": "system", "content": full_system}]
    for m in conversation_messages[-12:]:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=messages,
        )
        text = (resp.choices[0].message.content if resp.choices else "") or ""
        return text.strip()
    except Exception as e:
        logger.exception("OpenAI API chat failed: %s", e)
        raise


def _call_kimi_simple(prompt: str, model: str, *, max_tokens: int = 2048) -> str:
    """Kimi K2.5 via OpenAI-compatible API at https://api.moonshot.cn/v1."""
    from config.llm_config import get_llm_config
    from openai import OpenAI
    cfg = get_llm_config()
    if not cfg.moonshot_api_key:
        raise ValueError("MOONSHOT_API_KEY not set for Kimi.")
    try:
        client = OpenAI(api_key=cfg.moonshot_api_key, base_url="https://api.moonshot.cn/v1")
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp.choices[0].message.content if resp.choices else "") or ""
        return text.strip()
    except Exception as e:
        logger.exception("Kimi API call failed: %s", e)
        raise


def _call_kimi_chat(
    system_prompt: str,
    context_block: str,
    conversation_messages: List[Dict[str, str]],
    user_message: str,
    model: str,
    *,
    max_tokens: int = 2048,
) -> str:
    from config.llm_config import get_llm_config
    from openai import OpenAI
    cfg = get_llm_config()
    if not cfg.moonshot_api_key:
        raise ValueError("MOONSHOT_API_KEY not set for Kimi.")
    try:
        client = OpenAI(api_key=cfg.moonshot_api_key, base_url="https://api.moonshot.cn/v1")
        full_system = f"{system_prompt}\n\n{context_block}"
        messages = [{"role": "system", "content": full_system}]
        for m in conversation_messages[-12:]:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=messages,
        )
        text = (resp.choices[0].message.content if resp.choices else "") or ""
        return text.strip()
    except Exception as e:
        logger.exception("Kimi API chat failed: %s", e)
        raise


def _call_gemini_simple(prompt: str, model: str, *, max_tokens: int = 2048) -> str:
    from config.llm_config import get_llm_config
    cfg = get_llm_config()
    if not cfg.google_api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set for Gemini.")
    try:
        import google.generativeai as genai
        genai.configure(api_key=cfg.google_api_key)
        m = genai.GenerativeModel(model)
        r = m.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=max_tokens, temperature=0.3))
        return (r.text or "").strip()
    except Exception as e:
        logger.exception("Gemini API call failed: %s", e)
        raise


def _call_gemini_chat(
    system_prompt: str,
    context_block: str,
    conversation_messages: List[Dict[str, str]],
    user_message: str,
    model: str,
    *,
    max_tokens: int = 2048,
) -> str:
    from config.llm_config import get_llm_config
    cfg = get_llm_config()
    if not cfg.google_api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set for Gemini.")
    full_system = f"{system_prompt}\n\n{context_block}"
    parts = [full_system, "\n\nConversation:"]
    for m in conversation_messages[-12:]:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role}: {content}")
    parts.append(f"user: {user_message}")
    parts.append("assistant:")
    single_prompt = "\n".join(parts)
    return _call_gemini_simple(single_prompt, model, max_tokens=max_tokens)


def _call_ollama_simple(prompt: str, model: str, *, max_tokens: int = 2048) -> str:
    import urllib.request
    import json
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.3},
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return (data.get("response") or "").strip()
    except urllib.error.URLError as e:
        logger.exception("Ollama request failed (is Ollama running at http://localhost:11434?): %s", e)
        raise
    except Exception as e:
        logger.exception("Ollama API call failed: %s", e)
        raise


def _call_ollama_chat(
    system_prompt: str,
    context_block: str,
    conversation_messages: List[Dict[str, str]],
    user_message: str,
    model: str,
    *,
    max_tokens: int = 2048,
) -> str:
    import urllib.request
    import json
    full_system = f"{system_prompt}\n\n{context_block}"
    url = "http://localhost:11434/api/chat"
    messages = [{"role": "system", "content": full_system}]
    for m in conversation_messages[-12:]:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_message})
    payload = {"model": model, "messages": messages, "stream": False, "options": {"num_predict": max_tokens, "temperature": 0.3}}
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return (data.get("message", {}).get("content") or "").strip()
    except urllib.error.URLError as e:
        logger.exception("Ollama chat request failed (is Ollama running at http://localhost:11434?): %s", e)
        raise
    except Exception as e:
        logger.exception("Ollama API chat failed: %s", e)
        raise


def _call_huggingface_simple(prompt: str, model: str, *, max_tokens: int = 2048) -> str:
    from config.llm_config import get_llm_config
    cfg = get_llm_config()
    api_key = cfg.huggingface_api_key or ""
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY or HF_TOKEN not set for HuggingFace Inference API.")
    import urllib.request
    import json
    url = f"https://api-inference.huggingface.co/models/{model}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": 0.3, "return_full_text": False}}
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict) and "generated_text" in data[0]:
            return (data[0]["generated_text"] or "").strip()
        if isinstance(data, dict) and "generated_text" in data:
            return (data["generated_text"] or "").strip()
        return ""
    except Exception as e:
        logger.exception("HuggingFace Inference API call failed: %s", e)
        raise


def _call_huggingface_chat(
    system_prompt: str,
    context_block: str,
    conversation_messages: List[Dict[str, str]],
    user_message: str,
    model: str,
    *,
    max_tokens: int = 2048,
) -> str:
    full_system = f"{system_prompt}\n\n{context_block}"
    combined = full_system + "\n\n"
    for m in conversation_messages[-12:]:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        combined += f"{role}: {content}\n\n"
    combined += f"user: {user_message}\n\nassistant:"
    return _call_huggingface_simple(combined, model, max_tokens=max_tokens)


# Cache for HuggingFace local pipeline (avoid reload every request)
_hf_local_pipeline_cache: Dict[str, Any] = {}


def _call_huggingface_local_simple(prompt: str, model: str, *, max_tokens: int = 2048) -> str:
    """HuggingFace local pipeline via transformers (fully offline)."""
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError as e:
        raise ValueError("transformers not installed. pip install transformers") from e
    global _hf_local_pipeline_cache
    if model not in _hf_local_pipeline_cache:
        try:
            _hf_local_pipeline_cache[model] = hf_pipeline("text-generation", model=model, device=-1)
        except Exception as e:
            logger.exception("HuggingFace local pipeline load failed: %s", e)
            raise
    pipe = _hf_local_pipeline_cache[model]
    try:
        out = pipe(prompt, max_new_tokens=max_tokens, temperature=0.3, do_sample=True, return_full_text=False)
        if isinstance(out, list) and len(out) > 0:
            first = out[0]
            text = first.get("generated_text", "") if isinstance(first, dict) else str(first)
            return (text or "").strip()
        return ""
    except Exception as e:
        logger.exception("HuggingFace local generation failed: %s", e)
        raise


def _call_huggingface_local_chat(
    system_prompt: str,
    context_block: str,
    conversation_messages: List[Dict[str, str]],
    user_message: str,
    model: str,
    *,
    max_tokens: int = 2048,
) -> str:
    full_system = f"{system_prompt}\n\n{context_block}"
    combined = full_system + "\n\n"
    for m in conversation_messages[-12:]:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        combined += f"{role}: {content}\n\n"
    combined += f"user: {user_message}\n\nassistant:"
    return _call_huggingface_local_simple(combined, model, max_tokens=max_tokens)


def call_active_llm_simple(prompt: str, *, max_tokens: int = 2048) -> str:
    """
    Single prompt → response using the app's active LLM (from Admin preference).
    Raises on failure; no silent fallback.
    Routes to all six providers: claude, gpt4, gemini, ollama, kimi, huggingface.
    """
    # Ensure env has latest keys from user store if missing
    try:
        from config.user_store import inject_user_keys_to_env
        from streamlit import session_state as st_session_state  # type: ignore
        session_id = st_session_state.get("evolve_session_id", "") or st_session_state.get("session_id", "")
        if session_id:
            inject_user_keys_to_env(str(session_id))
    except Exception:
        pass

    env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    provider, model, options = get_active_llm()
    if env_provider in {"openai", "anthropic", "huggingface"}:
        logger.info("LLM_PROVIDER override active: %s", env_provider)
        if env_provider == "anthropic":
            return _call_claude_simple(prompt, model, max_tokens=max_tokens)
        if env_provider == "openai":
            return _call_gpt4_simple(prompt, model, max_tokens=max_tokens)
        if env_provider == "huggingface":
            hf_mode = options.get("huggingface_mode", "inference")
            if hf_mode == "local":
                return _call_huggingface_local_simple(prompt, model, max_tokens=max_tokens)
            return _call_huggingface_simple(prompt, model, max_tokens=max_tokens)

    if provider == "claude":
        try:
            return _call_claude_simple(prompt, model, max_tokens=max_tokens)
        except Exception as e:
            if _is_anthropic_401(e):
                logger.warning("Anthropic key invalid, falling back to OpenAI")
                try:
                    from config.llm_config import DEFAULT_MODELS
                    openai_model = DEFAULT_MODELS.get("gpt4", "gpt-4o")
                except Exception:
                    openai_model = "gpt-4o"
                return _call_gpt4_simple(prompt, openai_model, max_tokens=max_tokens)
            raise
    if provider == "gpt4":
        return _call_gpt4_simple(prompt, model, max_tokens=max_tokens)
    if provider == "gemini":
        return _call_gemini_simple(prompt, model, max_tokens=max_tokens)
    if provider == "ollama":
        return _call_ollama_simple(prompt, model, max_tokens=max_tokens)
    if provider == "kimi":
        return _call_kimi_simple(prompt, model, max_tokens=max_tokens)
    if provider == "huggingface":
        hf_mode = options.get("huggingface_mode", "inference")
        if hf_mode == "local":
            return _call_huggingface_local_simple(prompt, model, max_tokens=max_tokens)
        return _call_huggingface_simple(prompt, model, max_tokens=max_tokens)
    raise ValueError(f"Unknown active provider: {provider}")


def call_active_llm_chat(
    system_prompt: str,
    context_block: str,
    conversation_messages: List[Dict[str, str]],
    user_message: str,
    *,
    max_tokens: int = 2048,
) -> str:
    """
    Chat (system + context + history + user message) using the app's active LLM.
    Raises on failure; no silent fallback.
    Routes to all six providers: claude, gpt4, gemini, ollama, kimi, huggingface.
    """
    try:
        from config.user_store import inject_user_keys_to_env
        from streamlit import session_state as st_session_state  # type: ignore
        session_id = st_session_state.get("evolve_session_id", "") or st_session_state.get("session_id", "")
        if session_id:
            inject_user_keys_to_env(str(session_id))
    except Exception:
        pass

    env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    provider, model, options = get_active_llm()
    if env_provider in {"openai", "anthropic", "huggingface"}:
        logger.info("LLM_PROVIDER override active: %s", env_provider)
        if env_provider == "anthropic":
            return _call_claude_chat(system_prompt, context_block, conversation_messages, user_message, model, max_tokens=max_tokens)
        if env_provider == "openai":
            return _call_gpt4_chat(system_prompt, context_block, conversation_messages, user_message, model, max_tokens=max_tokens)
        if env_provider == "huggingface":
            hf_mode = options.get("huggingface_mode", "inference")
            if hf_mode == "local":
                return _call_huggingface_local_chat(system_prompt, context_block, conversation_messages, user_message, model, max_tokens=max_tokens)
            return _call_huggingface_chat(system_prompt, context_block, conversation_messages, user_message, model, max_tokens=max_tokens)

    if provider == "claude":
        return _call_claude_chat(system_prompt, context_block, conversation_messages, user_message, model, max_tokens=max_tokens)
    if provider == "gpt4":
        return _call_gpt4_chat(system_prompt, context_block, conversation_messages, user_message, model, max_tokens=max_tokens)
    if provider == "gemini":
        return _call_gemini_chat(system_prompt, context_block, conversation_messages, user_message, model, max_tokens=max_tokens)
    if provider == "ollama":
        return _call_ollama_chat(system_prompt, context_block, conversation_messages, user_message, model, max_tokens=max_tokens)
    if provider == "kimi":
        return _call_kimi_chat(system_prompt, context_block, conversation_messages, user_message, model, max_tokens=max_tokens)
    if provider == "huggingface":
        hf_mode = options.get("huggingface_mode", "inference")
        if hf_mode == "local":
            return _call_huggingface_local_chat(system_prompt, context_block, conversation_messages, user_message, model, max_tokens=max_tokens)
        return _call_huggingface_chat(system_prompt, context_block, conversation_messages, user_message, model, max_tokens=max_tokens)
    raise ValueError(f"Unknown active provider: {provider}")
