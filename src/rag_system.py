"""Recuperador híbrido: BM25 (léxico) + Pinecone (semántico) con EnsembleRetriever.

Uso:  python -m src.rag_system "¿Cómo habilito CORS para mi frontend?"
      python -m src.rag_system "¿Cómo manejo errores?" --categoria errores
      python -m src.rag_system "¿Cómo habilito CORS?" --json    (salida en JSON)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

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


@dataclass
class Fragmento:
    """Un chunk del top-k híbrido con los scores que explican su posición."""

    doc: Document
    score: float  # score RRF combinado: sum(peso / (posición + c))
    origen: list[str]  # recuperadores que lo trajeron: "bm25", "vector" o ambos
    similitud: float | None  # coseno contra la consulta (None si solo lo trajo BM25)


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
        min_similitud: float = config.MIN_SIMILITUD,
    ):
        self.k = k
        self.namespace = namespace
        self.min_similitud = min_similitud
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
        # (y sume score), en vez de compararlos por texto. La fusión se hace con
        # su weighted_reciprocal_rank sobre rankings ya filtrados (por metadata
        # y descartando lo irrelevante), con los mismos pesos, c e id_key.
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

        Devuelve una lista vacía si la pregunta no es del dominio (ver
        `retrieve_con_scores`).
        """
        fragmentos, _ = self.retrieve_con_scores(query, filtro)
        return [f.doc for f in fragmentos]

    def retrieve_bm25(self, query: str, filtro: dict | None = None) -> list[Document]:
        # BM25 corre en memoria: se puntúa todo el corpus y se descartan los
        # chunks que no cumplen el filtro antes de cortar el top-k. También los
        # de score 0 (ningún término en común con la consulta): el retriever de
        # LangChain los devuelve igual para completar el top-k, y en una
        # pregunta fuera de tema eso es puro relleno.
        scores = self.bm25_retriever.vectorizer.get_scores(bm25_tokenize(query))
        candidatos = [
            (score, doc)
            for score, doc in zip(scores, self.bm25_retriever.docs)
            if score > 0 and cumple_filtro(doc.metadata, filtro)
        ]
        candidatos.sort(key=lambda par: par[0], reverse=True)
        return [doc for _, doc in candidatos[: self.k]]

    def retrieve_vector(self, query: str, filtro: dict | None = None) -> list[Document]:
        return [doc for doc, _ in self._vector_con_scores(query, filtro)]

    def _vector_con_scores(self, query: str, filtro: dict | None = None) -> list[tuple[Document, float]]:
        # Pinecone aplica el filtro del lado del servidor, dentro del namespace,
        # y devuelve la similitud coseno de cada chunk.
        return self.vectorstore.similarity_search_with_score(query, k=self.k, filter=filtro)

    def retrieve_con_scores(
        self, query: str, filtro: dict | None = None
    ) -> tuple[list[Fragmento], float]:
        """Top-k híbrido con el score combinado de cada chunk, y la similitud máxima.

        El score es el que el EnsembleRetriever calcula internamente y no
        expone: sum(peso / (posición + c)) sobre cada ranking en el que aparece
        el chunk.

        Un recuperador top-k siempre devuelve k resultados, aunque la pregunta
        no tenga nada que ver con el corpus. Si la similitud coseno del mejor
        chunk vectorial queda por debajo de `min_similitud`, se considera que
        la pregunta está fuera del dominio y se devuelve una lista vacía.
        """
        vectoriales = self._vector_con_scores(query, filtro)
        similitud_max = max((s for _, s in vectoriales), default=0.0)
        if similitud_max < self.min_similitud:
            return [], similitud_max

        rankings = {"bm25": self.retrieve_bm25(query, filtro), "vector": [d for d, _ in vectoriales]}
        scores: dict[str, float] = defaultdict(float)
        origen: dict[str, list[str]] = defaultdict(list)
        for (nombre, docs), peso in zip(rankings.items(), self.ensemble.weights):
            for rank, doc in enumerate(docs, start=1):
                cid = doc.metadata["chunk_id"]
                scores[cid] += peso / (rank + self.ensemble.c)
                origen[cid].append(nombre)
        similitud = {d.metadata["chunk_id"]: s for d, s in vectoriales}
        fusion = self.ensemble.weighted_reciprocal_rank(list(rankings.values()))[: self.k]
        fragmentos = []
        for d in fusion:
            cid = d.metadata["chunk_id"]
            fragmentos.append(Fragmento(d, scores[cid], origen[cid], similitud.get(cid)))
        return fragmentos, similitud_max

def _extracto(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rstrip() + "..."


def resultado_json(
    rag: RAGSystem, query: str, filtro: dict | None, resultados: list[Fragmento], similitud_max: float
) -> dict:
    """Salida estructurada de una consulta: metadata de cada chunk + score combinado."""
    fragmentos = [
        {
            "chunk_id": d.metadata["chunk_id"],
            "doc_id": d.metadata.get("doc_id"),
            "fuente": d.metadata.get("source"),
            "categoria": d.metadata.get("category"),
            "seccion": d.metadata.get("section"),
            # Pinecone devuelve los números de la metadata como float (2.0).
            "page": int(d.metadata["page"]) if d.metadata.get("page") is not None else None,
            "score_combinado": round(f.score, 4),
            "similitud_coseno": round(f.similitud, 3) if f.similitud is not None else None,
            "recuperado_por": f.origen,
            "extracto": _extracto(d.page_content, 200),
        }
        for f in resultados
        for d in [f.doc]
    ]
    salida = {
        "pregunta": query,
        "estrategia_recuperacion": "hibrida_ensemble (bm25 + pinecone_dense, RRF)",
        "pesos": {"bm25": rag.ensemble.weights[0], "vector": rag.ensemble.weights[1]},
        "indice": config.INDEX_NAME,
        "namespace": rag.namespace,
        "filtro": filtro,
        "top_k": rag.k,
        "similitud_maxima": round(similitud_max, 3),
        "umbral_similitud": rag.min_similitud,
        "fragmentos_recuperados": fragmentos,
        "fuentes": list(dict.fromkeys(f["fuente"] for f in fragmentos)),
    }
    if not fragmentos:
        salida["mensaje"] = (
            f"Sin resultados relevantes: la similitud máxima ({similitud_max:.3f}) está por debajo "
            f"del umbral ({rag.min_similitud}). La pregunta parece estar fuera de la documentación indexada."
        )
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta el recuperador híbrido.")
    parser.add_argument("query")
    parser.add_argument("--categoria", help='atajo para --filtro \'{"category": "<categoria>"}\'')
    parser.add_argument("--filtro", help='filtro de metadata en JSON (sintaxis de Pinecone)')
    parser.add_argument("--json", action="store_true", help="imprime el resultado como JSON")
    args = parser.parse_args()
    config.setup_logging()

    filtro = json.loads(args.filtro) if args.filtro else None
    if args.categoria:
        filtro = {**(filtro or {}), "category": args.categoria}

    rag = RAGSystem()
    salida = resultado_json(rag, args.query, filtro, *rag.retrieve_con_scores(args.query, filtro))
    if args.json:
        print(json.dumps(salida, ensure_ascii=False, indent=2))
        return

    print(f"\nConsulta: {args.query}")
    print(
        f"Índice: {salida['indice']} · namespace: {salida['namespace']} · "
        f"estrategia: híbrida (BM25 + Pinecone, RRF, pesos {rag.ensemble.weights})"
    )
    if filtro:
        print(f"Filtro de metadata: {json.dumps(filtro, ensure_ascii=False)}")
    if not salida["fragmentos_recuperados"]:
        print(salida["mensaje"])
        return
    print(f"Top-{rag.k} (similitud coseno máxima {salida['similitud_maxima']:.3f}, umbral {rag.min_similitud}):")
    for rank, f in enumerate(salida["fragmentos_recuperados"], start=1):
        print(
            f"  {rank}. {f['chunk_id']:<24} score={f['score_combinado']:.4f}  [{f['categoria']}]  "
            f"{f['fuente']} · page {f['page']}  (de: {'+'.join(f['recuperado_por'])})"
        )
        print(f"     {f['seccion'][:45]!r}: {_extracto(f['extracto'], 90)}")


if __name__ == "__main__":
    main()
