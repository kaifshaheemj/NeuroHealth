import json

from groq import Groq

import config

_client = None


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set. "
                "Set it as an environment variable or in config.py"
            )
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def call_groq_chat(
    system_prompt: str,
    messages: list[dict],
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Non-streaming chat completion. Returns assistant content string."""
    model = model or config.GROQ_MODEL
    client = get_groq_client()

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def call_groq_streaming(
    system_prompt: str,
    messages: list[dict],
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
):
    """Streaming chat completion. Yields content chunks."""
    model = model or config.GROQ_MODEL
    client = get_groq_client()

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    stream = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def call_groq_json(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    temperature: float = 0.1,
) -> dict:
    """Call Groq expecting JSON output. Parses and returns dict."""
    model = model or config.GROQ_FAST_MODEL
    client = get_groq_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}
