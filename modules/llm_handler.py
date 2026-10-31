"""
modules/llm_handler.py
=======================
Thin wrapper around the Ollama REST API.

Responsibilities:
  - Health-check (is Ollama running?).
  - Model inventory (which models are installed?).
  - Detect whether a model supports vision (multimodal).
  - Text generation (chat completions).
  - Vision-language generation (image + text prompt).
  - Configurable timeout and streaming support.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests

from modules.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_VISION_MODEL

logger = logging.getLogger(__name__)

# Known vision-capable model families (lowercase prefix match)
_VISION_MODEL_FAMILIES = frozenset({
    "llava", "bakllava", "moondream", "llava-phi3", "llava-llama3",
    "minicpm-v", "cogvlm", "internvl", "qwen2-vl", "qwen2.5-vl",
    "gemma3", "granite3.2-vision", "llama3.2-vision",
})


# ──────────────────────────────────────────────────────────────────────────────
# Status / probe helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_ollama_running(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Return True if the Ollama server responds to a health ping."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def list_installed_models(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    """Return a list of installed model names."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        logger.warning("Could not list Ollama models: %s", exc)
        return []


def model_supports_vision(model_name: str, base_url: str = OLLAMA_BASE_URL) -> bool:
    """
    Heuristic + API check: does this model support image inputs?

    Checks:
      1. Known vision model family name prefix.
      2. Ollama /api/show model details (projector / modality field).
    """
    name_lower = model_name.lower()
    # Check family prefix
    for family in _VISION_MODEL_FAMILIES:
        if name_lower.startswith(family):
            return True

    # Try the show endpoint for detailed capability info
    try:
        r = requests.post(
            f"{base_url}/api/show",
            json={"name": model_name},
            timeout=10,
        )
        if r.status_code == 200:
            details = r.json()
            # Ollama returns "vision" in model_info.general.architecture or details
            model_info = details.get("model_info", {})
            arch = str(model_info.get("general.architecture", "")).lower()
            if "vision" in arch or "clip" in arch or "projector" in str(details).lower():
                return True
    except Exception:
        pass

    return False


def ollama_status(
    text_model: str = OLLAMA_MODEL,
    vision_model: str = OLLAMA_VISION_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> Dict[str, Any]:
    """Comprehensive status dict for the settings/dashboard page."""
    running = is_ollama_running(base_url)
    if not running:
        return {
            "running": False,
            "text_model_available": False,
            "vision_model_available": False,
            "installed_models": [],
            "error": "Ollama is not running. Start it with: ollama serve",
        }

    installed = list_installed_models(base_url)
    return {
        "running": True,
        "installed_models": installed,
        "text_model": text_model,
        "text_model_available": text_model in installed,
        "vision_model": vision_model,
        "vision_model_available": vision_model in installed if vision_model else False,
        "vision_supported": model_supports_vision(vision_model, base_url) if vision_model in installed else False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Text generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_text(
    prompt: str,
    system_prompt: str = "",
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: int = OLLAMA_TIMEOUT,
    stream: bool = False,
) -> str:
    """
    Send a text prompt to Ollama and return the full response.

    Raises:
        RuntimeError: if Ollama is unreachable or returns an error.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        r = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it is running: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama request timed out after {timeout}s. "
            "Try a smaller model or increase OLLAMA_TIMEOUT."
        )
    except Exception as exc:
        raise RuntimeError(f"Ollama text generation failed: {exc}") from exc


def generate_text_chat(
    messages: List[Dict[str, str]],
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: int = OLLAMA_TIMEOUT,
) -> str:
    """Multi-turn chat generation with a list of {role, content} messages."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Run: ollama serve")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama request timed out after {timeout}s.")
    except Exception as exc:
        raise RuntimeError(f"Ollama chat failed: {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
# Vision-language generation
# ──────────────────────────────────────────────────────────────────────────────

def _encode_image_b64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_image(
    image_path: str,
    context_prompt: str = "Describe this image in detail, noting all visual elements, diagrams, charts, or text visible.",
    model: str = OLLAMA_VISION_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = OLLAMA_TIMEOUT,
) -> Tuple[str, bool]:
    """
    Send an image to a VLM and return (description, success).

    Returns:
        (description_text, True)  on success
        (error_message,   False)  on failure
    """
    if not model:
        return "Vision model not configured.", False

    if not Path(image_path).exists():
        return f"Image file not found: {image_path}", False

    if not model_supports_vision(model, base_url):
        return (
            f"Model '{model}' does not appear to support image inputs. "
            "Configure OLLAMA_VISION_MODEL with a vision-capable model (e.g. llava:7b).",
            False,
        )

    try:
        img_b64 = _encode_image_b64(image_path)
    except Exception as exc:
        return f"Failed to read image: {exc}", False

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": context_prompt,
                "images": [img_b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1024},
    }

    try:
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        description = r.json()["message"]["content"].strip()
        return description, True
    except requests.exceptions.ConnectionError:
        return "Ollama is not running.", False
    except requests.exceptions.Timeout:
        return f"VLM request timed out after {timeout}s.", False
    except Exception as exc:
        return f"Vision generation failed: {exc}", False


def answer_about_image(
    image_path: str,
    question: str,
    model: str = OLLAMA_VISION_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = OLLAMA_TIMEOUT,
) -> Tuple[str, bool]:
    """Ask a specific question about a single image."""
    prompt = (
        f"Answer the following question about the image: {question}\n\n"
        "Be specific and reference what is visible in the image."
    )
    return describe_image(
        image_path=image_path,
        context_prompt=prompt,
        model=model,
        base_url=base_url,
        timeout=timeout,
    )
