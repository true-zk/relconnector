"""Export the frozen GloVe tokenizer and weights for native batch encoding."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, cast

DEFAULT_GLOVE_MODEL = (
    "sentence-transformers/average_word_embeddings_glove.6B.300d"
)


def ensure_glove_export(
    model_path: str | None = None,
    *,
    directory: Path | None = None,
) -> Path:
    """Create a local native artifact once; batch encoding stays in C++."""
    source = model_path or DEFAULT_GLOVE_MODEL
    if directory is None:
        name = hashlib.sha256(source.encode()).hexdigest()[:16]
        directory = Path.home() / ".cache" / "relx" / f"glove-{name}"
    directory = directory.expanduser().resolve()
    manifest = directory / "manifest.json"
    if manifest.is_file():
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        weights = directory / "weights.f32"
        vocabulary = directory / "vocab.bin"
        if (
            metadata.get("source") == source
            and metadata.get("format") == 1
            and weights.is_file()
            and vocabulary.is_file()
            and weights.stat().st_size
            == metadata.get("rows", 0) * metadata.get("width", 0) * 4
        ):
            return directory

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Exporting GloVe requires sentence-transformers"
        ) from exc

    model = SentenceTransformer(source, device="cpu", local_files_only=True)
    modules = list(model)
    if len(modules) != 2 or cast(Any, modules[1]).get_pooling_mode_str() != "mean":
        raise ValueError("Native GloVe needs word embeddings with mean pooling")
    word = cast(Any, modules[0])
    tokenizer = cast(Any, word.tokenizer)
    if not hasattr(tokenizer, "word2idx"):
        raise ValueError("Native GloVe needs the whitespace tokenizer")
    vocabulary_words = list(tokenizer.word2idx)
    weights_array = word.emb_layer.weight.detach().cpu().contiguous().numpy()
    rows, width = weights_array.shape
    if rows != len(vocabulary_words) or str(weights_array.dtype) != "float32":
        raise ValueError("GloVe vocabulary and float32 weights disagree")
    stop_words = sorted(tokenizer.stop_words)

    directory.mkdir(parents=True, exist_ok=True)
    vocabulary_tmp = directory / "vocab.bin.tmp"
    weights_tmp = directory / "weights.f32.tmp"
    with vocabulary_tmp.open("wb") as output:
        output.write(struct.pack(
            "<IIIIII", 0x47584C52, 1, rows, width,
            int(tokenizer.do_lower_case), len(stop_words),
        ))
        for value in (*vocabulary_words, *stop_words):
            encoded = value.encode("utf-8")
            output.write(struct.pack("<I", len(encoded)))
            output.write(encoded)
    weights_array.tofile(weights_tmp)
    vocabulary_tmp.replace(directory / "vocab.bin")
    weights_tmp.replace(directory / "weights.f32")
    manifest.write_text(
        json.dumps({"format": 1, "source": source, "rows": rows, "width": width}),
        encoding="utf-8",
    )
    return directory
