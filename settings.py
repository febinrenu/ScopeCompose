"""Central configuration loader.

Two YAML files drive everything that could otherwise get hardcoded:

* ``config/hardware.yaml`` -- device, precision, batch sizes, model checkpoints.
  Read this rather than writing ``cuda`` or a batch size into a module, so the
  same code runs on Member B's machine and on a CPU-only CI box.
* ``config/models.yaml``  -- the logical-tier to concrete-model-ID map.
  Read this rather than naming a provider model in a module, so swapping
  providers is a config change.

Both are cached after first load. Call :func:`reload` in a test that needs to
change them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "config"

HARDWARE_YAML = CONFIG_DIR / "hardware.yaml"
MODELS_YAML = CONFIG_DIR / "models.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. It is part of the repo -- check out the file or "
            f"run scripts/discover_models.py if it is config/models.yaml."
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# --------------------------------------------------------------------------- #
# Hardware
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """Settings for one local model role."""

    model: str
    batch_size: int
    max_seq_length: int
    # fine-tuning only
    gradient_accumulation_steps: int = 1
    epochs: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class HardwareProfile:
    """A complete hardware profile.

    The default profile targets a 6 GB VRAM floor so shared code runs
    unmodified on both members' machines. Anything that needs more is opt-in.
    """

    name: str
    description: str
    device: str
    precision: str
    embeddings: ModelSpec
    cross_encoder: ModelSpec
    finetune: ModelSpec
    top_k: int
    rrf_k: int

    def resolve_device(self) -> str:
        """Turn ``device: auto`` into a concrete device string.

        Imports torch lazily so that modules which only need config -- the
        contract, the API layer, the metrics -- do not pay for a torch import.
        """
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def use_fp16(self) -> bool:
        return self.precision == "fp16" and self.resolve_device() != "cpu"


def _model_spec(raw: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        model=raw["model"],
        batch_size=int(raw["batch_size"]),
        max_seq_length=int(raw["max_seq_length"]),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 1)),
        epochs=int(raw.get("epochs", 1)),
        learning_rate=float(raw.get("learning_rate", 2e-5)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.1)),
    )


@lru_cache(maxsize=None)
def hardware(profile: str | None = None) -> HardwareProfile:
    """Load a hardware profile.

    Resolution order: explicit argument, then ``CONFLICT_RAG_HW_PROFILE``, then
    ``active_profile`` in the YAML.
    """
    raw = _load_yaml(HARDWARE_YAML)
    name = profile or os.environ.get("CONFLICT_RAG_HW_PROFILE") or raw.get("active_profile")

    profiles = raw.get("profiles", {})
    if name not in profiles:
        raise KeyError(
            f"hardware profile {name!r} not found in {HARDWARE_YAML.name}; "
            f"available: {sorted(profiles)}"
        )
    p = profiles[name]
    retrieval = raw.get("retrieval", {})

    return HardwareProfile(
        name=name,
        description=p.get("description", ""),
        device=p.get("device", "auto"),
        precision=p.get("precision", "fp16"),
        embeddings=_model_spec(p["embeddings"]),
        cross_encoder=_model_spec(p["cross_encoder"]),
        finetune=_model_spec(p["finetune"]),
        top_k=int(retrieval.get("top_k", 5)),
        rrf_k=int(retrieval.get("rrf_k", 60)),
    )


# --------------------------------------------------------------------------- #
# Models / tiers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str | None
    api_key: str | None

    @property
    def configured(self) -> bool:
        return bool(self.base_url) and bool(self.api_key)


@dataclass(frozen=True)
class TierSpec:
    """A logical model tier.

    Code asks for a tier; this maps it to a concrete model. The indirection is
    what keeps the zero-spend decision (README section 5) a one-line change if
    budget ever appears, rather than a refactor.
    """

    name: str
    provider: str
    model: str | None
    temperature: float
    max_tokens: int
    prefer: list[str]
    reasoning_effort: str | None = None
    """For reasoning models (gpt-oss on Groq): low | medium | high.

    Worth setting deliberately. Reasoning tokens are charged to the completion
    and count against both ``max_tokens`` and the daily cap, so on a capped
    free tier this is a budget control, not just a quality dial.
    """

    @property
    def resolved(self) -> bool:
        return self.model is not None


@dataclass(frozen=True)
class FallbackSpec:
    enabled: bool
    provider: str
    model: str | None
    trigger_after_retries: int


@dataclass(frozen=True)
class ModelsConfig:
    default_provider: str
    providers: dict[str, ProviderSpec]
    tiers: dict[str, TierSpec]
    fallback: FallbackSpec
    pricing: dict[str, dict[str, float]]

    def tier(self, name: str) -> TierSpec:
        key = name.upper()
        if key not in self.tiers:
            raise KeyError(f"unknown tier {name!r}; available: {sorted(self.tiers)}")
        return self.tiers[key]

    def provider(self, name: str) -> ProviderSpec:
        if name not in self.providers:
            raise KeyError(f"unknown provider {name!r}; available: {sorted(self.providers)}")
        return self.providers[name]

    def price(self, model: str) -> tuple[float, float]:
        """(input, output) USD per 1M tokens. Falls back to the default entry."""
        entry = self.pricing.get(model) or self.pricing.get("default", {})
        return float(entry.get("input_per_1m", 0.0)), float(entry.get("output_per_1m", 0.0))


@lru_cache(maxsize=None)
def models() -> ModelsConfig:
    raw = _load_yaml(MODELS_YAML)

    providers: dict[str, ProviderSpec] = {}
    for name, p in (raw.get("providers") or {}).items():
        base_url = os.environ.get(p["base_url_env"]) if p.get("base_url_env") else None
        if p.get("api_key_env"):
            api_key = os.environ.get(p["api_key_env"])
        else:
            api_key = p.get("api_key_literal")
        providers[name] = ProviderSpec(name=name, base_url=base_url, api_key=api_key)

    tiers: dict[str, TierSpec] = {}
    for name, t in (raw.get("tiers") or {}).items():
        tiers[name.upper()] = TierSpec(
            name=name.upper(),
            provider=t.get("provider", raw.get("default_provider", "groq")),
            model=t.get("model"),
            temperature=float(t.get("temperature", 0.0)),
            max_tokens=int(t.get("max_tokens", 1024)),
            prefer=list(t.get("prefer") or []),
            reasoning_effort=t.get("reasoning_effort"),
        )

    fb = raw.get("fallback") or {}
    fallback = FallbackSpec(
        enabled=bool(fb.get("enabled", False)),
        provider=fb.get("provider", "ollama"),
        model=os.environ.get(fb["model_env"]) if fb.get("model_env") else fb.get("model"),
        trigger_after_retries=int(fb.get("trigger_after_retries", 3)),
    )

    return ModelsConfig(
        default_provider=raw.get("default_provider", "groq"),
        providers=providers,
        tiers=tiers,
        fallback=fallback,
        pricing=raw.get("pricing") or {},
    )


def reload() -> None:
    """Drop cached config. Call after editing a YAML or an env var in a test."""
    hardware.cache_clear()
    models.cache_clear()


def load_dotenv_if_present() -> None:
    """Load ``.env`` if python-dotenv is installed and the file exists.

    Kept optional so the package imports cleanly in CI, where secrets come from
    the environment rather than a file.
    """
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


load_dotenv_if_present()
