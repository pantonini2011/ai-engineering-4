from types import SimpleNamespace

import pytest
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.rag_system import RAGSystem, bm25_tokenize, cumple_filtro, load_corpus_from_pinecone


def test_tokenizer_conserva_identificadores_y_quita_stopwords():
    assert bm25_tokenize("¿Cómo uso `add_middleware` con CORSMiddleware?") == [
        "uso", "add_middleware", "corsmiddleware",
    ]


def test_tokenizer_ignora_tildes():
    assert bm25_tokenize("configuración") == bm25_tokenize("configuracion") == ["configuracion"]


def test_bm25_matchea_nombres_tecnicos_exactos():
    # Con 2 documentos el IDF de BM25Okapi para un término que aparece en 1
    # vale log(1.5/1.5) = 0: hace falta un corpus mínimamente realista.
    corpus = [
        Document(page_content="Usa HTTPException para devolver errores", metadata={"chunk_id": "a"}),
        Document(page_content="Declara un request body con Pydantic", metadata={"chunk_id": "b"}),
        Document(page_content="Agrega CORSMiddleware para permitir orígenes", metadata={"chunk_id": "c"}),
        Document(page_content="Ejecuta tareas en segundo plano después del response", metadata={"chunk_id": "d"}),
    ]
    bm25 = BM25Retriever.from_documents(corpus, preprocess_func=bm25_tokenize, k=1)
    assert bm25.invoke("¿cómo lanzo una HTTPException?")[0].metadata["chunk_id"] == "a"


class FakeIndex:
    """Imita index.list() (páginas de IDs) e index.fetch() de Pinecone."""

    def __init__(self, vectors):
        self.vectors = vectors

    def list(self, namespace):
        ids = list(self.vectors)
        yield ids[:1]
        yield ids[1:]

    def fetch(self, ids, namespace):
        return SimpleNamespace(
            vectors={i: SimpleNamespace(metadata=self.vectors[i]) for i in ids}
        )


def test_corpus_bm25_se_reconstruye_desde_metadata_de_pinecone():
    index = FakeIndex({
        "cors#002": {"text": "dos", "chunk_id": "cors#002", "doc_id": "cors"},
        "cors#001": {"text": "uno", "chunk_id": "cors#001", "doc_id": "cors"},
    })
    docs = load_corpus_from_pinecone(index, namespace="dev")
    assert [d.page_content for d in docs] == ["uno", "dos"]
    assert "text" not in docs[0].metadata and docs[0].metadata["doc_id"] == "cors"


# --- Filtros de metadata ---------------------------------------------------

META = {"category": "seguridad", "doc_id": "cors", "page": 2.0}


@pytest.mark.parametrize(
    "filtro, esperado",
    [
        (None, True),
        ({}, True),
        ({"category": "seguridad"}, True),
        ({"category": "errores"}, False),
        ({"doc_id": {"$in": ["cors", "middleware"]}}, True),
        ({"doc_id": {"$nin": ["cors"]}}, False),
        ({"page": {"$lte": 2}}, True),
        ({"page": {"$gt": 2}}, False),
        ({"category": "seguridad", "page": {"$gte": 3}}, False),
        ({"$or": [{"category": "errores"}, {"doc_id": "cors"}]}, True),
        ({"$and": [{"category": "seguridad"}, {"doc_id": {"$ne": "cors"}}]}, False),
        ({"inexistente": {"$gt": 1}}, False),
    ],
)
def test_cumple_filtro_sintaxis_pinecone(filtro, esperado):
    assert cumple_filtro(META, filtro) is esperado


def test_cumple_filtro_rechaza_operador_desconocido():
    with pytest.raises(ValueError, match="no soportado"):
        cumple_filtro(META, {"page": {"$regex": "x"}})


def _doc(cid, category, text):
    return Document(
        page_content=text,
        metadata={"chunk_id": cid, "doc_id": cid.split("#")[0], "category": category},
    )


CORPUS = [
    _doc("cors#001", "seguridad", "CORSMiddleware permite origenes y headers"),
    _doc("security#001", "seguridad", "OAuth2 y headers de autenticacion"),
    _doc("handling-errors#001", "errores", "HTTPException devuelve errores y headers personalizados"),
    _doc("middleware#001", "infraestructura", "Un middleware procesa cada request y agrega headers"),
    _doc("body#001", "parametros", "Declara un request body con Pydantic"),
]


class FakeVectorStore:
    """Sustituye a PineconeVectorStore: sin filtro devuelve el corpus en orden;
    con filtro, lo aplica como haría Pinecone del lado del servidor."""

    def __init__(self, docs):
        self.docs = docs
        self.filtros_recibidos = []

    def as_retriever(self, search_kwargs):
        return RunnableLambda(lambda q: self.docs[: search_kwargs["k"]])

    def similarity_search(self, query, k, filter):
        self.filtros_recibidos.append(filter)
        return [d for d in self.docs if cumple_filtro(d.metadata, filter)][:k]


def _rag(k=3):
    return RAGSystem(k=k, corpus=CORPUS, vectorstore=FakeVectorStore(CORPUS))


def test_retrieve_sin_filtro_usa_todo_el_corpus():
    categorias = {d.metadata["category"] for d in _rag().retrieve("headers")}
    assert len(categorias) > 1


def test_retrieve_con_filtro_solo_devuelve_chunks_que_cumplen():
    rag = _rag()
    docs = rag.retrieve("headers", filtro={"category": "seguridad"})
    assert {d.metadata["chunk_id"] for d in docs} == {"cors#001", "security#001"}
    assert rag.vectorstore.filtros_recibidos == [{"category": "seguridad"}]


def test_bm25_filtra_antes_de_cortar_el_top_k():
    # Sin filtro, "HTTPException" pone a handling-errors primero; con filtro de
    # seguridad el top-k se llena igual con chunks de seguridad (no queda vacío).
    rag = _rag(k=1)
    assert rag.retrieve_bm25("HTTPException headers")[0].metadata["doc_id"] == "handling-errors"
    (doc,) = rag.retrieve_bm25("HTTPException headers", filtro={"category": "seguridad"})
    assert doc.metadata["category"] == "seguridad"


def test_filtro_sin_coincidencias_devuelve_lista_vacia():
    assert _rag().retrieve("headers", filtro={"category": "inexistente"}) == []
