"""Recuperador híbrido: BM25 (léxico) + Pinecone (semántico) con EnsembleRetriever.

Uso:  python -m src.rag_system "¿Cómo habilito CORS para mi frontend?"
      python -m src.rag_system "¿Cómo manejo errores?" --categoria errores
"""

from __future__ import annotations

import argparse
import json
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


_OPERADORES = {
    "$eq": lambda v, x: v == x,
    "$ne": lambda v, x: v != x,
    "$gt": lambda v, x: v is not None and v > x,
    "$gte": lambda v, x: v is not None and v >= x,
    "$lt": lambda v, x: v is not None and v < x,
    "$lte": lambda v, x: v is not None and v <= x,
    "$in": lambda v, x: v in x,
    "$nin": lambda v, x: v not in x,
}


def cumple_filtro(metadata: dict, filtro: dict | None) -> bool:
    """Evalúa un filtro con la sintaxis de metadata de Pinecone sobre un dict local.

    Soporta `{"campo": valor}` (igualdad implícita), los operadores `$eq`, `$ne`,
    `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin` y la combinación con `$and` /
    `$or`. Es la misma sintaxis que Pinecone aplica del lado del servidor en la
    búsqueda vectorial, así un único filtro sirve para los dos recuperadores.
    """
    if not filtro:
        return True
    for campo, cond in filtro.items():
        if campo == "$and":
            ok = all(cumple_filtro(metadata, f) for f in cond)
        elif campo == "$or":
            ok = any(cumple_filtro(metadata, f) for f in cond)
        elif isinstance(cond, dict):
            valor = metadata.get(campo)
            for op, x in cond.items():
                if op not in _OPERADORES:
                    raise ValueError(f"Operador de filtro no soportado: {op}")
            ok = all(_OPERADORES[op](valor, x) for op, x in cond.items())
        else:
            ok = metadata.get(campo) == cond
        if not ok:
            return False
    return True


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
        vectorstore=None,
    ):
        self.k = k
        self.namespace = namespace
        if corpus is None or vectorstore is None:
            index = index or ensure_index()

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
        self.vectorstore = vectorstore or PineconeVectorStore(
            index=index, embedding=embeddings or build_embeddings(),
            text_key="text", namespace=namespace,
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

    def retrieve(self, query: str, filtro: dict | None = None) -> list[Document]:
        """Top-k híbrido (léxico + semántico), opcionalmente filtrado por metadata.

        `filtro` usa la sintaxis de Pinecone, ej. `{"category": "seguridad"}` o
        `{"doc_id": {"$in": ["cors", "middleware"]}}`. Se aplica a los dos
        recuperadores *antes* de rankear: así el top-k sale completo del
        subconjunto filtrado, en vez de filtrar un top-k ya recortado.
        """
        if not filtro:
            return self.ensemble.invoke(query)[: self.k]
        # Mismos pesos, id_key y RRF que el EnsembleRetriever, sobre los
        # rankings ya filtrados.
        rankings = [self.retrieve_bm25(query, filtro), self.retrieve_vector(query, filtro)]
        return self.ensemble.weighted_reciprocal_rank(rankings)[: self.k]

    def retrieve_bm25(self, query: str, filtro: dict | None = None) -> list[Document]:
        if not filtro:
            return self.bm25_retriever.invoke(query)[: self.k]
        # BM25 corre en memoria: se puntúa todo el corpus y se descartan los
        # chunks que no cumplen el filtro antes de cortar el top-k.
        scores = self.bm25_retriever.vectorizer.get_scores(bm25_tokenize(query))
        candidatos = [
            (score, doc)
            for score, doc in zip(scores, self.bm25_retriever.docs)
            if cumple_filtro(doc.metadata, filtro)
        ]
        candidatos.sort(key=lambda par: par[0], reverse=True)
        return [doc for _, doc in candidatos[: self.k]]

    def retrieve_vector(self, query: str, filtro: dict | None = None) -> list[Document]:
        if not filtro:
            return self.vector_retriever.invoke(query)[: self.k]
        # Pinecone aplica el filtro del lado del servidor, dentro del namespace.
        return self.vectorstore.similarity_search(query, k=self.k, filter=filtro)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta el recuperador híbrido.")
    parser.add_argument("query")
    parser.add_argument("--categoria", help='atajo para --filtro \'{"category": "<categoria>"}\'')
    parser.add_argument("--filtro", help='filtro de metadata en JSON (sintaxis de Pinecone)')
    args = parser.parse_args()
    config.setup_logging()

    filtro = json.loads(args.filtro) if args.filtro else None
    if args.categoria:
        filtro = {**(filtro or {}), "category": args.categoria}

    rag = RAGSystem()
    bm25_ids = {d.metadata["chunk_id"] for d in rag.retrieve_bm25(args.query, filtro)}
    vec_ids = {d.metadata["chunk_id"] for d in rag.retrieve_vector(args.query, filtro)}

    print(f"\nConsulta: {args.query}")
    if filtro:
        print(f"Filtro de metadata: {json.dumps(filtro, ensure_ascii=False)}")
    print(f"Top-{rag.k} híbrido (BM25 + Pinecone):")
    for rank, doc in enumerate(rag.retrieve(args.query, filtro), start=1):
        cid = doc.metadata["chunk_id"]
        origen = "+".join(n for n, s in (("bm25", bm25_ids), ("vector", vec_ids)) if cid in s)
        print(
            f"  {rank}. {cid:<26} [{doc.metadata['category']}] "
            f"{doc.metadata['section'][:45]!r}  (de: {origen})"
        )


if __name__ == "__main__":
    main()
