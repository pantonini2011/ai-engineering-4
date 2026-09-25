"""Recuperador híbrido: BM25 (léxico) + Pinecone (semántico) con EnsembleRetriever.

Uso:  python -m src.rag_system "How do I enable CORS for my frontend?"
"""

from __future__ import annotations

import argparse
import logging
import re

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from src import config
from src.ingestion import build_embeddings
from src.setup_index import ensure_index

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
# Palabras vacías del inglés (el idioma del corpus). Sin filtrarlas, BM25
# premia coincidencias en "how", "do", "the"... y trae chunks de otros temas.
_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in into is it its "
    "my of on or so that the their then there these this to use using want was "
    "we what when where which while will with without you your".split()
)


def bm25_tokenize(text: str) -> list[str]:
    """Tokenizer de BM25: minúsculas, palabras alfanuméricas, sin stopwords.

    Conserva el guion bajo para que identificadores como `add_middleware` o
    `response_model` queden como un solo término (la gracia de BM25 es
    matchear nombres técnicos exactos). El default de LangChain (`split()`)
    dejaría pegada la puntuación: "`HTTPException`," != "HTTPException".
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def load_corpus_from_pinecone(index, namespace: str = config.NAMESPACE) -> list[Document]:
    """Reconstruye los chunks leyendo el texto guardado en la metadata de Pinecone.

    BM25 necesita el corpus completo en memoria para calcular IDF. En vez de
    re-leer y re-fragmentar los archivos locales (que podrían haber cambiado
    desde la ingesta), se toma lo que realmente está indexado: así BM25 y el
    buscador vectorial ven exactamente el mismo conjunto de chunks.
    """
    docs: list[Document] = []
    for ids in index.list(namespace=namespace):
        fetched = index.fetch(ids=list(ids), namespace=namespace)
        for vid, vec in fetched.vectors.items():
            meta = dict(vec.metadata or {})
            text = meta.pop("text", "")
            docs.append(Document(page_content=text, metadata=meta, id=vid))
    docs.sort(key=lambda d: d.metadata.get("chunk_id", d.id))
    return docs


class RAGSystem:
    """Encapsula un EnsembleRetriever (BM25 + Pinecone) que devuelve el top-k."""

    def __init__(
        self,
        k: int = config.TOP_K,
        weights: list[float] | None = None,
        namespace: str = config.NAMESPACE,
        index=None,
        embeddings=None,
        corpus: list[Document] | None = None,
    ):
        self.k = k
        self.namespace = namespace
        index = index or ensure_index()
        embeddings = embeddings or build_embeddings()

        corpus = corpus if corpus is not None else load_corpus_from_pinecone(index, namespace)
        if not corpus:
            raise RuntimeError(
                f"El namespace '{namespace}' del índice '{config.INDEX_NAME}' está vacío. "
                "Corré primero: python -m src.ingestion"
            )
        logger.info("Corpus BM25: %d chunks leídos de Pinecone (namespace '%s')", len(corpus), namespace)

        self.bm25_retriever = BM25Retriever.from_documents(
            corpus, preprocess_func=bm25_tokenize, k=k
        )
        self.vectorstore = PineconeVectorStore(
            index=index, embedding=embeddings, text_key="text", namespace=namespace
        )
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        # Fusiona ambos rankings con Reciprocal Rank Fusion. id_key hace que un
        # mismo chunk traído por los dos recuperadores cuente una sola vez
        # (y sume score), en vez de compararlos por texto.
        self.ensemble = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=weights or config.ENSEMBLE_WEIGHTS,
            id_key="chunk_id",
        )

    def retrieve(self, query: str) -> list[Document]:
        """Top-k híbrido (léxico + semántico)."""
        return self.ensemble.invoke(query)[: self.k]

    def retrieve_bm25(self, query: str) -> list[Document]:
        return self.bm25_retriever.invoke(query)[: self.k]

    def retrieve_vector(self, query: str) -> list[Document]:
        return self.vector_retriever.invoke(query)[: self.k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta el recuperador híbrido.")
    parser.add_argument("query")
    args = parser.parse_args()
    config.setup_logging()

    rag = RAGSystem()
    bm25_ids = {d.metadata["chunk_id"] for d in rag.retrieve_bm25(args.query)}
    vec_ids = {d.metadata["chunk_id"] for d in rag.retrieve_vector(args.query)}

    print(f"\nConsulta: {args.query}\nTop-{rag.k} híbrido (BM25 + Pinecone):")
    for rank, doc in enumerate(rag.retrieve(args.query), start=1):
        cid = doc.metadata["chunk_id"]
        origen = "+".join(n for n, s in (("bm25", bm25_ids), ("vector", vec_ids)) if cid in s)
        print(
            f"  {rank}. {cid:<26} [{doc.metadata['category']}] "
            f"{doc.metadata['section'][:45]!r}  (de: {origen})"
        )


if __name__ == "__main__":
    main()
