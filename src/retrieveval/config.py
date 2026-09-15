"""Benchmark configuration, loaded from YAML.

Validation is hand-written rather than delegated to a schema library so that
every failure can name the offending key and say what was expected -- the CLI
promises useful errors, and "chunking.overlap must be smaller than
chunking.size" is more actionable than a schema trace.

Unknown keys are rejected: a typo in a config file would otherwise be silently
ignored and produce a benchmark that does not measure what the user asked for.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

RETRIEVER_NAMES = ("bm25", "dense", "hybrid")

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CACHE_DIR = ".retrieveval_cache"


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or self-contradictory."""


# --------------------------------------------------------------------------
# small validation helpers -- each one names the key it was reading
# --------------------------------------------------------------------------


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping, got {type(value).__name__}")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: Iterable[str], where: str) -> None:
    # Keys are not necessarily strings: YAML reads `2024:` as an int and, under
    # YAML 1.1 rules, `on:` and `yes:` as booleans. Comparing and sorting those
    # against string keys would raise TypeError and escape the CLI's handler,
    # so everything is normalised to text before being reported.
    permitted = set(allowed)
    unknown = sorted(str(key) for key in data if key not in permitted)
    if unknown:
        options = ", ".join(sorted(allowed))
        raise ConfigError(
            f"unknown key{'s' if len(unknown) > 1 else ''} in {where}: "
            f"{', '.join(unknown)} (allowed: {options})"
        )


def _positive_int(data: Mapping[str, Any], key: str, where: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}.{key} must be an integer, got {value!r}")
    if value <= 0:
        raise ConfigError(f"{where}.{key} must be positive, got {value}")
    return value


def _non_negative_int(data: Mapping[str, Any], key: str, where: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}.{key} must be an integer, got {value!r}")
    if value < 0:
        raise ConfigError(f"{where}.{key} must not be negative, got {value}")
    return value


def _string(data: Mapping[str, Any], key: str, where: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}.{key} must be a non-empty string, got {value!r}")
    return value


def _bool(data: Mapping[str, Any], key: str, where: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{where}.{key} must be true or false, got {value!r}")
    return value


# --------------------------------------------------------------------------
# configuration sections
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkingConfig:
    size: int = 500
    overlap: int = 50

    @classmethod
    def parse(cls, raw: Any) -> ChunkingConfig:
        data = _mapping(raw, "chunking")
        _reject_unknown(data, ("size", "overlap"), "chunking")
        size = _positive_int(data, "size", "chunking", cls.size)
        overlap = _non_negative_int(data, "overlap", "chunking", cls.overlap)
        if overlap >= size:
            raise ConfigError(
                "chunking.overlap must be smaller than chunking.size "
                f"(got overlap={overlap}, size={size}); an overlap at or above the "
                "chunk size would never advance through the document"
            )
        return cls(size=size, overlap=overlap)


@dataclass(frozen=True)
class DenseConfig:
    model: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 32
    cache_dir: str = DEFAULT_CACHE_DIR

    @classmethod
    def parse(cls, raw: Any) -> DenseConfig:
        data = _mapping(raw, "dense")
        _reject_unknown(data, ("model", "batch_size", "cache_dir"), "dense")
        return cls(
            model=_string(data, "model", "dense", cls.model),
            batch_size=_positive_int(data, "batch_size", "dense", cls.batch_size),
            cache_dir=_string(data, "cache_dir", "dense", cls.cache_dir),
        )


@dataclass(frozen=True)
class HybridConfig:
    rrf_k: int = 60
    """The RRF smoothing constant. 60 is the value from the original paper;
    larger values flatten the contribution of top ranks."""
    depth: int = 100
    """How many candidates to take from each retriever before fusing. Fusing
    only the final top-k would discard the very agreements RRF exists to find."""

    @classmethod
    def parse(cls, raw: Any) -> HybridConfig:
        data = _mapping(raw, "hybrid")
        _reject_unknown(data, ("rrf_k", "depth"), "hybrid")
        return cls(
            rrf_k=_positive_int(data, "rrf_k", "hybrid", cls.rrf_k),
            depth=_positive_int(data, "depth", "hybrid", cls.depth),
        )


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    model: str = DEFAULT_RERANKER_MODEL
    candidates: int = 50
    """How many results the base retriever hands to the cross-encoder."""
    applies_to: tuple[str, ...] = ()
    """Base retrievers to produce a reranked variant of. Empty means all of them."""

    @classmethod
    def parse(cls, raw: Any, retrievers: Sequence[str]) -> RerankerConfig:
        data = _mapping(raw, "reranker")
        _reject_unknown(data, ("enabled", "model", "candidates", "applies_to"), "reranker")
        enabled = _bool(data, "enabled", "reranker", cls.enabled)
        applies_to = _parse_retriever_list(
            data.get("applies_to", list(retrievers)), "reranker.applies_to"
        )
        unconfigured = [name for name in applies_to if name not in retrievers]
        if enabled and unconfigured:
            raise ConfigError(
                f"reranker.applies_to names retriever(s) that are not configured: "
                f"{', '.join(unconfigured)}; add them to `retrievers` or remove them here"
            )
        return cls(
            enabled=enabled,
            model=_string(data, "model", "reranker", cls.model),
            candidates=_positive_int(data, "candidates", "reranker", cls.candidates),
            applies_to=applies_to,
        )


@dataclass(frozen=True)
class EvaluationConfig:
    k: tuple[int, ...] = (1, 5, 10)

    @classmethod
    def parse(cls, raw: Any) -> EvaluationConfig:
        data = _mapping(raw, "evaluation")
        _reject_unknown(data, ("k",), "evaluation")
        values = data.get("k", list(cls.k))
        if isinstance(values, int) and not isinstance(values, bool):
            values = [values]
        if not isinstance(values, Sequence) or isinstance(values, str) or not values:
            raise ConfigError(f"evaluation.k must be a non-empty list of integers, got {values!r}")
        cutoffs = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(
                    f"evaluation.k entries must be positive integers, got {value!r}"
                )
            cutoffs.append(value)
        return cls(k=tuple(sorted(set(cutoffs))))

    @property
    def max_k(self) -> int:
        return max(self.k)


def _parse_retriever_list(raw: Any, where: str) -> tuple[str, ...]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Sequence) or not raw:
        raise ConfigError(
            f"{where} must be a non-empty list of retriever names "
            f"({', '.join(RETRIEVER_NAMES)}), got {raw!r}"
        )
    names: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            raise ConfigError(f"{where} entries must be strings, got {entry!r}")
        name = entry.strip().lower()
        if name not in RETRIEVER_NAMES:
            raise ConfigError(
                f"unknown retriever {entry!r} in {where}; "
                f"supported retrievers are: {', '.join(RETRIEVER_NAMES)}"
            )
        if name not in names:
            names.append(name)
    return tuple(names)


@dataclass(frozen=True)
class Config:
    """A complete benchmark configuration."""

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrievers: tuple[str, ...] = RETRIEVER_NAMES
    dense: DenseConfig = field(default_factory=DenseConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    TOP_LEVEL_KEYS = ("chunking", "retrievers", "dense", "hybrid", "reranker", "evaluation")

    @classmethod
    def from_mapping(cls, raw: Any) -> Config:
        data = _mapping(raw, "configuration root")
        _reject_unknown(data, cls.TOP_LEVEL_KEYS, "the configuration file")
        retrievers = _parse_retriever_list(
            data.get("retrievers", list(RETRIEVER_NAMES)), "retrievers"
        )
        return cls(
            chunking=ChunkingConfig.parse(data.get("chunking")),
            retrievers=retrievers,
            dense=DenseConfig.parse(data.get("dense")),
            hybrid=HybridConfig.parse(data.get("hybrid")),
            reranker=RerankerConfig.parse(data.get("reranker"), retrievers),
            evaluation=EvaluationConfig.parse(data.get("evaluation")),
        )

    @classmethod
    def load(cls, path: str | Path) -> Config:
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigError(f"configuration file not found: {config_path}")
        if not config_path.is_file():
            raise ConfigError(f"configuration path is not a file: {config_path}")
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"{config_path} is not valid YAML: {exc}") from exc
        try:
            return cls.from_mapping(raw)
        except ConfigError as exc:
            raise ConfigError(f"{config_path}: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """A plain-data view, recorded alongside results for reproducibility."""
        return asdict(self) | {"retrievers": list(self.retrievers)}

    @property
    def needs_dense_model(self) -> bool:
        return "dense" in self.retrievers or "hybrid" in self.retrievers
