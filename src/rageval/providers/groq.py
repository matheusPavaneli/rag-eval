import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from rageval.providers.base import ChatResult, PermanentProviderError, Usage, post_json

NAME = "groq"
CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class _Message(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    content: str = ""


class _Choice(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    message: _Message = _Message()


class _Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    input_tokens: int = Field(default=0, alias="prompt_tokens")
    output_tokens: int = Field(default=0, alias="completion_tokens")


class _ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    choices: tuple[_Choice, ...]
    usage: _Usage = _Usage()


class GroqChatProvider:
    def __init__(
        self, client: httpx.Client, api_key: SecretStr, model: str, timeout: float
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return NAME

    @property
    def model(self) -> str:
        return self._model

    def complete(self, prompt: str, system: str | None = None) -> ChatResult:
        messages: list[dict[str, str]] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body = post_json(
            self._client,
            CHAT_URL,
            headers={
                "authorization": f"Bearer {self._api_key.get_secret_value()}",
                "content-type": "application/json",
            },
            payload={"model": self._model, "messages": messages, "temperature": 0},
            timeout=self._timeout,
            provider=NAME,
            model=self._model,
        )

        try:
            parsed = _ChatResponse.model_validate(body)
        except ValidationError as error:
            raise PermanentProviderError(
                f"{NAME} {self._model}: response did not hold the expected fields"
            ) from error

        if not parsed.choices:
            raise PermanentProviderError(f"{NAME} {self._model}: response held no choice")

        return ChatResult(
            provider=NAME,
            model=self._model,
            text=parsed.choices[0].message.content,
            usage=Usage(
                input_tokens=parsed.usage.input_tokens,
                output_tokens=parsed.usage.output_tokens,
            ),
        )
