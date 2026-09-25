"""Pipeline de ingesta: Markdown -> chunks -> embeddings -> Pinecone.

Uso:
    python -m src.ingestion            # ingesta idempotente (omite si ya está poblado)
    python -m src.ingestion --reset    # vacía el namespace y re-indexa todo

Esquema de cada vector en Pinecone:
    id        "<doc_id>#<nnn>" (determinístico: re-ingestar pisa, no duplica)
    values    embedding de 1536 dims (text-embedding-3-small)
    metadata  text, doc_id, chunk_id, source, page, section, category,
              n_tokens, env, created_at

El texto original va en `metadata.text`: al recuperar no hace falta ir a
buscar el contenido a otra base (relacional o de archivos).
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from pinecone.exceptions import NotFoundException, PineconeException

from src import config
from src.setup_index import ensure_index

logger = logging.getLogger(__name__)

# Etiqueta de categoría por documento (metadato filtrable, ej. {"category": "seguridad"}).
CATEGORIES: dict[str, str] = {
    "first-steps": "fundamentos",
    "path-params": "parametros",
    "query-params": "parametros",
    "body": "parametros",
    "dependencies": "arquitectura",
    "bigger-applications": "arquitectura",
    "security": "seguridad",
    "cors": "seguridad",
    "handling-errors": "errores",
    "middleware": "infraestructura",
    "background-tasks": "infraestructura",
    "sql-databases": "persistencia",
}

_ENCODING = tiktoken.get_encoding("cl100k_base")
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def clean_markdown(text: str) -> str:
    """Quita ruido propio de la doc de FastAPI que no aporta al significado.

    - anclas de encabezado: `## Origin { #origin }` -> `## Origin`
    - includes de código externo: `{* ../../docs_src/x.py hl[6] *}` (apuntan a
      archivos que no están en el corpus; embebidos solo meten ruido)
    """
    text = re.sub(r"\s*\{\s*#[\w-]+\s*\}", "", text)
    text = re.sub(r"^\{\*.*?\*\}\s*$", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_documents(docs_dir: Path = config.DOCS_DIR) -> list[Document]:
    docs = []
    for path in sorted(docs_dir.glob("*.md")):
        doc_id = path.stem
        docs.append(
            Document(
                page_content=clean_markdown(path.read_text(encoding="utf-8")),
                metadata={
                    "doc_id": doc_id,
                    "source": f"data/docs/{path.name}",
                    "category": CATEGORIES.get(doc_id, "general"),
                },
            )
        )
    if not docs:
        raise SystemExit(f"No hay archivos .md en {docs_dir}")
    return docs


def _section_at(text: str, offset: int) -> str:
    """Último encabezado Markdown que aparece antes de `offset`."""
    section = ""
    for m in _HEADER_RE.finditer(text):
        if m.start() > offset:
            break
        section = m.group(2)
    return section


def split_documents(docs: list[Document]) -> list[Document]:
    """Chunking con RecursiveCharacterTextSplitter medido en tokens.

    Los separadores de `Language.MARKDOWN` priorizan cortar en encabezados,
    luego en bloques de código y párrafos, así cada chunk tiende a quedar
    dentro de una misma sección.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        separators=RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN),
        # Los separadores de Markdown son regex (ej. "\n#{1,6} "): sin esto se
        # buscarían literalmente y nunca cortaría en los encabezados.
        is_separator_regex=True,
        add_start_index=True,
    )
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    chunks: list[Document] = []
    for doc in docs:
        for i, chunk in enumerate(splitter.split_documents([doc]), start=1):
            start = chunk.metadata.pop("start_index")
            if start < 0:  # el splitter no ubicó el chunk exacto (overlap/strip)
                start = doc.page_content.find(chunk.page_content[:100].strip())
            chunk_id = f"{doc.metadata['doc_id']}#{i:03d}"
            chunk.metadata.update(
                {
                    "chunk_id": chunk_id,
                    # Markdown no tiene páginas físicas: "page" es la posición
                    # del chunk dentro del documento (1, 2, 3...).
                    "page": i,
                    "section": _section_at(doc.page_content, start) or doc.metadata["doc_id"],
                    "n_tokens": count_tokens(chunk.page_content),
                    "env": config.NAMESPACE,
                    "created_at": created_at,
                }
            )
            chunks.append(chunk)
    return chunks


def build_embeddings() -> OpenAIEmbeddings:
    """Único lugar donde se crea el modelo de embeddings (ingesta y consulta)."""
    config.require("OPENAI_API_KEY")
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


def _with_retries(fn, what: str):
    """Reintenta errores transitorios (red, rate limit) con backoff exponencial."""
    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            return fn()
        except PineconeException as exc:
            if attempt == config.MAX_RETRIES:
                logger.error("%s falló tras %d intentos: %s", what, attempt, exc)
                raise
            wait = 2 ** attempt
            logger.warning("%s falló (intento %d): %s. Reintento en %ds", what, attempt, exc, wait)
            time.sleep(wait)


def upsert_chunks(index, chunks: list[Document], vectors: list[list[float]]) -> int:
    if any(len(v) != config.EMBEDDING_DIM for v in vectors):
        raise ValueError(f"Algún embedding no tiene dimensión {config.EMBEDDING_DIM}")

    records = [
        {
            "id": c.metadata["chunk_id"],
            "values": v,
            "metadata": {"text": c.page_content, **c.metadata},
        }
        for c, v in zip(chunks, vectors)
    ]
    total = 0
    for start in range(0, len(records), config.UPSERT_BATCH_SIZE):
        batch = records[start : start + config.UPSERT_BATCH_SIZE]
        _with_retries(
            lambda: index.upsert(vectors=batch, namespace=config.NAMESPACE),
            f"upsert lote {start // config.UPSERT_BATCH_SIZE + 1}",
        )
        total += len(batch)
        logger.info("Upsert lote %d: %d vectores", start // config.UPSERT_BATCH_SIZE + 1, len(batch))
    return total


def namespace_count(index) -> int:
    stats = index.describe_index_stats()
    ns = (stats.namespaces or {}).get(config.NAMESPACE)
    return ns.vector_count if ns else 0


def wait_for_count(index, expected: int, timeout: float = 60.0) -> int:
    """Pinecone Serverless es eventualmente consistente: espera a ver los vectores."""
    deadline = time.monotonic() + timeout
    count = namespace_count(index)
    while count != expected and time.monotonic() < deadline:
        time.sleep(2)
        count = namespace_count(index)
    return count


def ingest(reset: bool = False) -> int:
    index = ensure_index()
    docs = load_documents()
    chunks = split_documents(docs)
    tokens = [c.metadata["n_tokens"] for c in chunks]
    logger.info(
        "%d documentos -> %d chunks (tokens por chunk: min=%d, prom=%d, max=%d)",
        len(docs), len(chunks), min(tokens), sum(tokens) // len(tokens), max(tokens),
    )

    existing = namespace_count(index)
    if reset and existing:
        logger.info("--reset: borrando %d vectores del namespace '%s'", existing, config.NAMESPACE)
        try:
            index.delete(delete_all=True, namespace=config.NAMESPACE)
        except NotFoundException:
            pass
        wait_for_count(index, 0)
    elif existing == len(chunks):
        logger.info(
            "Namespace '%s' ya tiene %d vectores (= chunks actuales): se omite la "
            "re-ingesta. Usá --reset para forzarla.", config.NAMESPACE, existing,
        )
        return 0

    embeddings = build_embeddings()
    logger.info("Generando %d embeddings con %s...", len(chunks), config.EMBEDDING_MODEL)
    vectors = embeddings.embed_documents([c.page_content for c in chunks])
    total = upsert_chunks(index, chunks, vectors)

    final = wait_for_count(index, len(chunks))
    logger.info(
        "Ingesta completa: %d vectores subidos; namespace '%s' del índice '%s' tiene %d.",
        total, config.NAMESPACE, config.INDEX_NAME, final,
    )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true", help="vacía el namespace antes de indexar")
    args = parser.parse_args()
    config.setup_logging()
    ingest(reset=args.reset)


if __name__ == "__main__":
    main()
