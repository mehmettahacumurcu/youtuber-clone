"""Strict immutable public contracts for the loopback voice service."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


VoiceStatus = Literal[
    "idle", "loading_xtts", "synthesizing", "loading_rvc", "converting", "maintenance", "error"
]

# The Studio permits up to 4,096 generated tokens. Reserving eight Unicode
# characters per token accommodates long Turkish prose without making the local
# synthesis endpoint unbounded.
MAX_SYNTHESIS_TEXT_CHARACTERS = 32_768

# Real packaged XTTS -> RVC synthesis is release-gated with this request bound.
# Keep the default and the accepted upper limit tied to the same contract.
MAX_VOICE_SYNTHESIS_TIMEOUT_SECONDS = 660.0
DEFAULT_VOICE_SYNTHESIS_TIMEOUT_SECONDS = MAX_VOICE_SYNTHESIS_TIMEOUT_SECONDS


class SynthesisRequest(BaseModel):
    """Caller input deliberately limited to natural text and an opaque request ID."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    text: StrictStr = Field(min_length=1, max_length=MAX_SYNTHESIS_TEXT_CHARACTERS)
    request_id: StrictStr = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")

    @field_validator("text")
    @classmethod
    def _nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class StatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: VoiceStatus


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ready"]
    runtime_api: Literal[1] = 1


class MaintenanceReleaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    lease_token: StrictStr = Field(min_length=32, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
