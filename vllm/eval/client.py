import aiohttp
from typing import List, Dict, Optional

from .config import CONFIG


def _payload(
    model_id: str,
    messages: List[Dict],
    *,
    temperature: float,
    max_tokens: int,
    seed: Optional[int] = None,
) -> Dict:
    p: Dict = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # For eval stability you usually want greedy decoding.
        "top_p": 1.0,
    }
    # llama-cpp-python server exposes a seed; other servers may ignore it.
    if seed is not None:
        p["seed"] = seed
    return p


async def _post_chat(
    endpoint: str,
    model_id: str,
    messages: List[Dict],
    timeout_s: int = 600,
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    seed: Optional[int] = 1234,
) -> str:
    url = f"{endpoint}/chat/completions"
    payload = _payload(
        model_id,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            # Don’t assume JSON if the server errored.
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"HTTP {resp.status} from {url}: {body[:500]}")

            data = await resp.json()

            if "error" in data:
                raise RuntimeError(data["error"])

            # OpenAI-compatible shape
            return data["choices"][0]["message"]["content"]


async def call_model(model_name: str, messages: List[Dict], timeout_s: int = 600) -> str:
    model_cfg = CONFIG["models"][model_name]
    # Allow per-model overrides from YAML/config if you add them later.
    return await _post_chat(
        model_cfg["endpoint"],
        model_cfg["model"],
        messages,
        timeout_s=timeout_s,
        temperature=model_cfg.get("temperature", 0.0),
        max_tokens=model_cfg.get("max_tokens", 512),
        seed=model_cfg.get("seed", 1234),
    )


async def call_judge(messages: List[Dict], timeout_s: int = 600) -> str:
    judge_cfg = CONFIG["judge"]
    return await _post_chat(
        judge_cfg["endpoint"],
        judge_cfg["model"],
        messages,
        timeout_s=timeout_s,
        temperature=judge_cfg.get("temperature", 0.0),
        max_tokens=judge_cfg.get("max_tokens", 512),
        seed=judge_cfg.get("seed", 1234),
    )
