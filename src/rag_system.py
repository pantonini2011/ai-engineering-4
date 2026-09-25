"""Recuperador híbrido: BM25 (léxico) + Pinecone (semántico) con EnsembleRetriever.

Uso:  python -m src.rag_system "¿Cómo habilito CORS para mi frontend?"
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from src import config
from src.ingestion import build_embeddings
from src.setup_index import ensure_index

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
# Palabras vacías del español (el idioma del corpus y de las preguntas), ya
# sin tildes porque se comparan después de normalizar. Sin filtrarlas, BM25
# premia coincidencias en "como", "de", "que"... y trae chunks de otros temas.
# Se suman algunas del inglés porque la doc mezcla términos en inglés
# ("request", "response") y comentarios de código.
_STOPWORDS = frozenset(
    # español
    "a al algo algun alguna algunas alguno algunos ante antes asi cada como con "
    "cual cuales cuando de del desde donde dos e el ella ellas ellos en entonces "
    "entre era es esa esas ese eso esos esta estan estas este esto estos fue ha "
    "hace hacer hacerlo hay la las le les lo los mas me mi mis mismo muy necesito "
    "ni no nos o otra otras otro otros para pero poder por porque puede pueden "
    "puedo que quiero se sea ser si sin sobre solo son su sus tambien tan te "
    "tener tengo tiene tu tus un una unas uno unos usar y ya yo "
    # inglés
    "a an and are as at be by do for from how i if in is it of on or that the "
    "this to with you your".split()
)


def _sin_tildes(text: str) -> str:
    """'configuración' -> 'configuracion' (y 'ñ' -> 'n'): tolera faltas de tildes."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def bm25_tokenize(text: str) -> list[str]:
    """Tokenizer de BM25: minúsculas, sin tildes, alfanuméricas, sin stopwords.

    Quitar tildes hace que "función" matchee "funcion" (y al revés): en una
    pregunta escrita rápido las tildes suelen faltar. Conserva el guion bajo para que identificadores como `add_middleware` o
    `response_model` queden como un solo término (la gracia de BM25 es
    matchear nombres técnicos exactos). El default de LangChain (`split()`)
    dejaría pegada la puntuación: "`HTTPException`," != "HTTPException".
    """
    return [t for t in _TOKEN_RE.findall(_sin_tildes(text.lower())) if t not in _STOPWORDS]


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
