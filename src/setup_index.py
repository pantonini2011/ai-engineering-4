"""Setup de Pinecone: verifica si el índice existe y lo crea si hace falta.

Uso:  python -m src.setup_index
"""

from __future__ import annotations

import logging
import time

from pinecone import Pinecone, ServerlessSpec

from src import config

logger = logging.getLogger(__name__)


def get_client() -> Pinecone:
    config.require("PINECONE_API_KEY")
    return Pinecone(api_key=config.PINECONE_API_KEY)


def ensure_index(pc: Pinecone | None = None, wait_timeout: float = 120.0):
    """Devuelve el índice `INDEX_NAME`, creándolo (Serverless) si no existe.

    Si el índice ya existe pero con otra dimensión o métrica, corta con un
    error en vez de dejar que la ingesta falle más tarde (o, peor, que se
    consulte un índice incompatible con el modelo de embeddings).
    """
    pc = pc or get_client()
    name = config.INDEX_NAME

    if name in pc.list_indexes().names():
        logger.info("El índice '%s' ya existe: no se recrea.", name)
    else:
        logger.info(
            "Creando índice Serverless '%s' (dim=%d, metric=%s, %s/%s)...",
            name, config.EMBEDDING_DIM, config.METRIC,
            config.PINECONE_CLOUD, config.PINECONE_REGION,
        )
        pc.create_index(
            name=name,
            dimension=config.EMBEDDING_DIM,
            metric=config.METRIC,
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )

    # Espera a que el índice esté listo (recién creado tarda unos segundos).
    deadline = time.monotonic() + wait_timeout
    desc = pc.describe_index(name)
    while not desc.status["ready"]:
        if time.monotonic() > deadline:
            raise TimeoutError(f"El índice '{name}' no quedó listo en {wait_timeout}s")
        time.sleep(2)
        desc = pc.describe_index(name)

    validate_index(desc)
    return pc.Index(name)


def validate_index(desc) -> None:
    """Chequea que dimensión y métrica del índice coincidan con la config."""
    if desc.dimension != config.EMBEDDING_DIM:
        raise ValueError(
            f"Mismatch de dimensiones: el índice '{desc.name}' tiene dimension="
            f"{desc.dimension}, pero {config.EMBEDDING_MODEL} genera vectores de "
            f"{config.EMBEDDING_DIM}. Usá otro INDEX_NAME o borrá el índice."
        )
    if desc.metric != config.METRIC:
        raise ValueError(
            f"El índice '{desc.name}' usa metric={desc.metric}; se esperaba {config.METRIC}."
        )
    logger.info(
        "Índice '%s' listo: dimension=%d, metric=%s, host=%s",
        desc.name, desc.dimension, desc.metric, desc.host,
    )


def main() -> None:
    config.setup_logging()
    index = ensure_index()
    stats = index.describe_index_stats()
    logger.info("Vectores totales: %d", stats.total_vector_count)
    for ns, info in (stats.namespaces or {}).items():
        logger.info("  namespace '%s': %d vectores", ns, info.vector_count)


if __name__ == "__main__":
    main()
