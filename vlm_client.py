import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional
from urllib import request

from openai import OpenAI

import config

logger = logging.getLogger("BioGazeAPI")


def _mime_type_for(path: Path) -> str:
    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }
    suffix = path.suffix.lower()
    if suffix not in mapping:
        raise ValueError(f"Unsupported image type: {suffix}")
    return mapping[suffix]


def _encode_image(path: Path) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{_mime_type_for(path)};base64,{encoded}"


def _build_messages(image_url: str, prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _normalize_yes_no(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    normalized = text.strip().lower()
    for candidate in (normalized, normalized.split()[0]):
        if candidate.startswith("yes"):
            return "yes"
        if candidate.startswith("no"):
            return "no"
    for token in normalized.replace(",", " ").split():
        if token in {"yes", "no"}:
            return token
    return None


def _strip_version_suffix(url: str) -> str:
    if not url:
        raise ValueError("VLM endpoint is empty")
    normalized = url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized.rstrip("/")


def _openai_base_url(endpoint: str) -> str:
    root = _strip_version_suffix(endpoint)
    return f"{root}/v1"


def ensure_vlm_alive(endpoint: str, timeout: float = 5.0) -> str:
    """
    Ping the VLM /ping endpoint to make sure the server is reachable.
    Returns the normalized root endpoint (without /v1) if successful.
    Raises RuntimeError on failure so callers can fail fast at startup.
    """
    root = _strip_version_suffix(endpoint)
    ping_url = f"{root}/ping"
    try:
        with request.urlopen(ping_url, timeout=timeout) as resp:
            status = getattr(resp, "status", None)
            if status != 200:
                raise RuntimeError(f"Unexpected ping status {status} from {ping_url}")
    except Exception as exc:
        raise RuntimeError(f"VLM server not reachable at {ping_url}: {exc}") from exc
    return root


class VLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: int,
        max_tokens: int,
        temperature: float,
        max_concurrency: int,
        ping_timeout: float = 5.0,
    ):
        # Fail fast if the VLM server is not reachable.
        self.endpoint_root = ensure_vlm_alive(base_url, timeout=ping_timeout)
        openai_base_url = _openai_base_url(self.endpoint_root)

        self.client = OpenAI(api_key=api_key, base_url=openai_base_url, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_concurrency = max(1, max_concurrency)

    def _send_request(self, image_url: str, prompt: str) -> Dict[str, object]:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=_build_messages(image_url, prompt),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        latency = time.perf_counter() - start
        content = response.choices[0].message.content if response.choices else ""
        answer = content.strip() if content else ""
        normalized = _normalize_yes_no(answer)
        return {
            "answer": answer,
            "normalized": normalized,
            "latency": latency,
        }

    def evaluate(self, image_path: str, prompts: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, object]]:
        if not config.VLM_ENABLED:
            return {}

        image_url = _encode_image(Path(image_path))
        results: Dict[str, Dict[str, object]] = {}

        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, max(1, len(prompts)))) as executor:
            future_map = {
                executor.submit(self._send_request, image_url, prompt_data["prompt"]): check_name
                for check_name, prompt_data in prompts.items()
            }
            for future in as_completed(future_map):
                check_name = future_map[future]
                try:
                    outcome = future.result()
                    results[check_name] = outcome
                    logger.info(
                        "VLM check '%s' response='%s' latency=%.2fs",
                        check_name,
                        outcome.get("answer", ""),
                        outcome.get("latency", 0.0),
                    )
                except Exception as exc:
                    logger.error("VLM check '%s' failed: %s", check_name, str(exc))
                    results[check_name] = {
                        "answer": None,
                        "normalized": None,
                        "latency": 0.0,
                        "error": str(exc),
                    }

        return results
