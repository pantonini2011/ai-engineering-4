"""Configuración central del proyecto (leída de .env).

Todo parámetro que tiene que coincidir entre la ingesta y la consulta
(modelo de embeddings, dimensión, índice, namespace) se define acá, en un
solo lugar, para evitar el error #1 de la consigna: el *mismatch* de
dimensiones entre los embeddings y el índice.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# --- Credenciales (obligatorias para hablar con Pinecone / OpenAI) ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Índice Pinecone Serverless ---
INDEX_NAME = os.getenv("INDEX_NAME", "fastapi-docs-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
# Un namespace por entorno: una re-ingesta de prueba en "dev" no pisa "prod".
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "dev")

# --- Embeddings ---
# text-embedding-3-small devuelve vectores de 1536 dimensiones normalizados
# (norma L2 = 1). Por eso el índice se crea con dimension=1536 y métrica
# coseno: con vectores unitarios, coseno mide solo la orientación (el tema)
# y no el largo del texto.
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))
METRIC = "cosine"

# --- Corpus y chunking ---
DOCS_DIR = ROOT_DIR / "data" / "docs"
GOLDEN_SET_PATH = ROOT_DIR / "data" / "golden_set.json"
# Tamaño medido en tokens (tiktoken cl100k_base, el tokenizer de
# text-embedding-3-small), dentro del rango 500-800 que sugiere la consigna.
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 80

# --- Ingesta ---
UPSERT_BATCH_SIZE = 100
MAX_RETRIES = 4

# --- Recuperación ---
TOP_K = 5
# Peso de cada recuperador en el EnsembleRetriever: [BM25, vectorial].
ENSEMBLE_WEIGHTS = [0.5, 0.5]


def setup_logging() -> None:
    # La consola de Windows usa cp1252 por default: forzar UTF-8 evita
    # acentos rotos y errores al redirigir la salida a un archivo.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silencia el log por request HTTP de los SDKs.
    for noisy in ("httpx", "httpx2", "openai", "pinecone", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def require(*names: str) -> None:
    """Corta con un mensaje claro si falta alguna variable obligatoria."""
    faltantes = [n for n in names if not os.getenv(n)]
    if faltantes:
        raise SystemExit(
            f"Faltan variables en .env: {', '.join(faltantes)}. "
            "Copiá .env.example a .env y completalas."
        )
