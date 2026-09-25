"""Discovery of the models already on this machine.

Two stores are scanned (see PROTOCOL.md):

1. ``$HF_HOME`` (default ``~/.cache/huggingface/hub``) for ``models--org--name``
   directories, whose snapshot is resolved through ``refs/main``;
2. ``$SLAM_LM_MODELS``, a colon-separated list of directories that contain
   ``org/name`` folders.

A directory qualifies only when it holds a ``config.json`` *and* at least one
weight file (``*.safetensors``, ``*.npz``, ``*.gguf``). Weight-less directories
are skipped, never listed as broken. Everything else — display name, parameter
count, quantisation and categories — is derived from real content only.
"""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .protocol import CATEGORY_ORDER, Model

#: Weight formats mlx-lm can actually load.
WEIGHT_SUFFIXES = (".safetensors", ".npz", ".gguf")

#: Hugging Face repos often keep the loadable model in a subfolder (e.g. a
#: multilingual encoder), so metadata and weights are searched a few levels in.
MAX_SEARCH_DEPTH = 4

QUANT_SUFFIX = re.compile(r"[-_](?:\d+\s*bit|fp16|bf16|fp32)(?:[-_].*)?$", re.IGNORECASE)

#: Architecture/model_type markers that mean "not a causal language model".
NON_CAUSAL_MARKERS = (
    "bert",
    "whisper",
    "wav2vec",
    "clip",
    "vit",
    "vision",
    "audio",
    "embedding",
    "speech",
    "onnx",
)

PARAM_BILLION = 1e9
PARAM_MILLION = 1e6
PARAM_THOUSAND = 1e3


# MARK: - Store locations


def default_hf_home() -> Path:
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser()
    return Path.home() / ".cache" / "huggingface" / "hub"


def extra_model_dirs() -> List[Path]:
    raw = os.environ.get("SLAM_LM_MODELS", "")
    return [Path(part).expanduser() for part in raw.split(":") if part.strip()]


# MARK: - Filesystem helpers


def _walk(root: Path, max_depth: int = MAX_SEARCH_DEPTH) -> Iterator[Path]:
    """Breadth-first walk of `root`, following directory symlinks once."""
    seen: set[str] = set()
    frontier: List[tuple[Path, int]] = [(root, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        try:
            real = os.path.realpath(current)
        except OSError:
            continue
        if real in seen:
            continue
        seen.add(real)
        yield current
        if depth >= max_depth:
            continue
        try:
            entries = sorted(os.scandir(current), key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    frontier.append((Path(entry.path), depth + 1))
            except OSError:
                continue


def _find_file(root: Path, filename: str) -> Optional[Path]:
    """Shallowest file named `filename` under `root` (root itself first)."""
    for directory in _walk(root):
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _weight_files(root: Path) -> List[Path]:
    """Weight files under `root`, deduplicated by their real path.

    The paths kept are the ones inside the snapshot (they carry the file
    extension used for the format check); symlinked blobs are followed for
    sizing and header reads, and counted only once.
    """
    found: Dict[str, Path] = {}
    for directory in _walk(root):
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            if not entry.name.lower().endswith(WEIGHT_SUFFIXES):
                continue
            try:
                if not entry.is_file():
                    continue
                real = os.path.realpath(entry.path)
                os.stat(real)  # must exist once symlinks are followed
            except OSError:
                continue
            if real not in found:
                found[real] = Path(entry.path)
    return [found[key] for key in sorted(found)]


def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


# MARK: - Derived fields


def display_name(model_id: str) -> str:
    """`mlx-community/Qwen3-1.7B-4bit` → `Qwen3 1.7B`; derived from the id."""
    base = model_id.split("/")[-1]
    base = QUANT_SUFFIX.sub("", base)
    base = base.replace("_", " ").replace("-", " ").strip()
    base = re.sub(r"\s+", " ", base)
    if not base:
        base = model_id.split("/")[-1]
    return base[0].upper() + base[1:] if base else model_id


def _quantization(config: Dict[str, Any]) -> tuple[str, Optional[int]]:
    block = config.get("quantization") or config.get("quantization_config")
    bits: Optional[int] = None
    if isinstance(block, dict):
        raw = block.get("bits")
        try:
            bits = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            bits = None
    if bits and bits > 0:
        return f"{bits}-bit", bits
    return "fp16", None


def _safetensors_header(path: Path) -> Optional[Dict[str, Any]]:
    """The JSON header of a .safetensors file: name → {dtype, shape, …}."""
    try:
        size = os.stat(path).st_size
        if size < 8:
            return None
        with open(path, "rb") as handle:
            (header_len,) = struct.unpack("<Q", handle.read(8))
            if header_len <= 0 or header_len > size - 8:
                return None
            raw = handle.read(header_len)
        header = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError, struct.error):
        return None
    return header if isinstance(header, dict) else None


def parameter_count(weights: Sequence[Path], bits: Optional[int]) -> Optional[int]:
    """Parameter count from the weight shapes, unpacking quantised weights.

    Quantised ``*.weight`` tensors are stored packed in ``uint32``; each element
    carries ``32 / bits`` parameters, which is exactly what the quantisation
    block says. Returns ``None`` when no readable shapes exist.
    """
    total = 0
    measured = False
    for path in weights:
        if path.suffix.lower() != ".safetensors":
            continue
        header = _safetensors_header(path)
        if header is None:
            continue
        measured = True
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            shape = meta.get("shape") or []
            elements = 1
            for dim in shape:
                try:
                    elements *= int(dim)
                except (TypeError, ValueError):
                    elements = 0
                    break
            if not elements:
                continue
            if (
                bits
                and meta.get("dtype") == "U32"
                and name.endswith(".weight")
                and (32 % bits) == 0
            ):
                elements *= 32 // bits
            total += elements
    return total if measured and total > 0 else None


def format_params(count: Optional[int]) -> str:
    """``""`` when unknown, otherwise a compact human count like ``1.7B``."""
    if not count:
        return ""
    if count >= PARAM_BILLION:
        return f"{count / PARAM_BILLION:.1f}B"
    if count >= PARAM_MILLION:
        return f"{count / PARAM_MILLION:.0f}M"
    if count >= PARAM_THOUSAND:
        return f"{count / PARAM_THOUSAND:.0f}K"
    return str(count)


def _is_causal_lm(config: Dict[str, Any], architectures: Sequence[str]) -> bool:
    for architecture in architectures:
        text = str(architecture)
        if "ForCausalLM" in text or "LMHeadModel" in text:
            return True
    markers = " ".join(str(a).lower() for a in architectures) + " " + str(
        config.get("model_type") or ""
    ).lower()
    return not any(marker in markers for marker in NON_CAUSAL_MARKERS)


def derive_categories(
    model_id: str, config: Dict[str, Any], has_chat_template: bool
) -> List[str]:
    """The category table from PROTOCOL.md, in canonical display order."""
    lowered = model_id.lower()
    architectures = config.get("architectures") or []
    if not isinstance(architectures, list):
        architectures = [architectures]
    architecture_text = " ".join(str(a) for a in architectures)

    found: set[str] = set()
    if has_chat_template:
        found.add("Instruct")
    if "vision_config" in config or "-vl" in lowered or "vision" in lowered:
        found.add("Vision")
    if (
        "Bert" in architecture_text
        or "Embedding" in architecture_text
        or "-embed" in lowered
    ):
        found.add("Embedding")
    if (
        "Whisper" in architecture_text
        or "Wav2Vec" in architecture_text
        or "-tts" in lowered
        or "-audio" in lowered
    ):
        found.add("Audio")
    if "code" in lowered or "coder" in lowered:
        found.add("Code")
    if "reasoning" in lowered or "r1" in lowered or "thinking" in lowered:
        found.add("Reasoning")
    try:
        vocab = int(config.get("vocab_size") or 0)
    except (TypeError, ValueError):
        vocab = 0
    if vocab >= 100_000:
        found.add("Multilingual")
    if _is_causal_lm(config, architectures):
        found.add("Chat")

    return [category for category in CATEGORY_ORDER if category in found]


def hello_categories(models: Sequence[Model]) -> List[str]:
    """Canonical display order, deduped, filtered to what was discovered."""
    discovered = {category for model in models for category in model.categories}
    return [category for category in CATEGORY_ORDER if category in discovered]


# MARK: - Snapshot resolution


def _snapshot_dir(model_dir: Path) -> Optional[Path]:
    """The loadable snapshot of a `models--org--name` directory."""
    refs = model_dir / "refs" / "main"
    try:
        revision = refs.read_text(encoding="utf-8").strip()
    except OSError:
        revision = ""
    if revision:
        candidate = model_dir / "snapshots" / revision
        if candidate.is_dir():
            return candidate.resolve()
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    try:
        candidates = [entry for entry in snapshots.iterdir() if entry.is_dir()]
    except OSError:
        return None
    if not candidates:
        return None
    newest = max(candidates, key=lambda entry: entry.stat().st_mtime)
    return newest.resolve()


def _candidates(
    hf_home: Optional[Path] = None, extra_dirs: Optional[Sequence[Path]] = None
) -> List[tuple[str, Path]]:
    """(repo id, directory to inspect) pairs from both stores."""
    out: List[tuple[str, Path]] = []
    seen: set[str] = set()

    home = Path(hf_home) if hf_home is not None else default_hf_home()
    if home.is_dir():
        try:
            entries = sorted(home.iterdir(), key=lambda entry: entry.name)
        except OSError:
            entries = []
        for entry in entries:
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue
            parts = entry.name.split("--", 2)
            if len(parts) != 3 or not parts[1] or not parts[2]:
                continue
            model_id = f"{parts[1]}/{parts[2]}"
            snapshot = _snapshot_dir(entry)
            if snapshot is None or model_id in seen:
                continue
            seen.add(model_id)
            out.append((model_id, snapshot))

    dirs = list(extra_dirs) if extra_dirs is not None else extra_model_dirs()
    for root in dirs:
        root = Path(root)
        if not root.is_dir():
            continue
        try:
            orgs = sorted(root.iterdir(), key=lambda entry: entry.name)
        except OSError:
            continue
        for org in orgs:
            if not org.is_dir() or org.name.startswith("."):
                continue
            try:
                names = sorted(org.iterdir(), key=lambda entry: entry.name)
            except OSError:
                continue
            for name in names:
                if not name.is_dir() or name.name.startswith("."):
                    continue
                model_id = f"{org.name}/{name.name}"
                if model_id in seen:
                    continue
                seen.add(model_id)
                out.append((model_id, name.resolve()))
    return out


# MARK: - Inspection


def inspect(
    model_id: str,
    root: Path,
    last_used: Optional[Dict[str, float]] = None,
) -> Optional[Model]:
    """Build a `Model` for one directory, or ``None`` when it does not qualify."""
    config_path = _find_file(root, "config.json")
    if config_path is None:
        return None  # no config: not a model, not listable
    weights = _weight_files(root)
    if not weights:
        return None  # weights-less directory: skipped entirely

    config = _read_json(config_path)
    tokenizer_config = _read_json(_find_file(root, "tokenizer_config.json"))

    quant, bits = _quantization(config)

    try:
        context_length = int(config.get("max_position_embeddings") or 0)
    except (TypeError, ValueError):
        context_length = 0

    architecture = str(config.get("model_type") or "")

    raw_template = tokenizer_config.get("chat_template")
    has_chat_template = isinstance(raw_template, str) and bool(raw_template.strip())

    try:
        total_bytes = sum(os.stat(path).st_size for path in weights)
    except OSError:
        total_bytes = 0

    used = 0.0
    if last_used:
        try:
            used = float(last_used.get(model_id, 0.0))
        except (TypeError, ValueError):
            used = 0.0

    return Model(
        id=model_id,
        name=display_name(model_id),
        params=format_params(parameter_count(weights, bits)),
        quant=quant,
        bytes=int(total_bytes),
        path=str(Path(root)),
        categories=derive_categories(model_id, config, has_chat_template),
        architecture=architecture,
        contextLength=context_length,
        hasChatTemplate=has_chat_template,
        lastUsed=used,
    )


def scan_models(
    hf_home: Optional[Path] = None,
    extra_dirs: Optional[Sequence[Path]] = None,
    last_used: Optional[Dict[str, float]] = None,
) -> List[Model]:
    """Every qualified local model, most recently used first."""
    models: List[Model] = []
    for model_id, root in _candidates(hf_home, extra_dirs):
        model = inspect(model_id, root, last_used)
        if model is not None:
            models.append(model)
    models.sort(key=lambda model: (-model.lastUsed, model.name.lower()))
    return models


def local_path(
    model_id: str,
    hf_home: Optional[Path] = None,
    extra_dirs: Optional[Sequence[Path]] = None,
) -> Optional[str]:
    """On-disk directory of a local model, or ``None`` when it is not local."""
    for candidate_id, root in _candidates(hf_home, extra_dirs):
        if candidate_id == model_id:
            return str(Path(root))
    return None
