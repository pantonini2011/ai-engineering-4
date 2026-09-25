from types import SimpleNamespace

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.rag_system import bm25_tokenize, load_corpus_from_pinecone


def test_tokenizer_conserva_identificadores_y_quita_stopwords():
    assert bm25_tokenize("How do I use `add_middleware`, CORSMiddleware?") == [
        "add_middleware", "corsmiddleware",
    ]


def test_bm25_matchea_nombres_tecnicos_exactos():
    # Con 2 documentos el IDF de BM25Okapi para un término que aparece en 1
    # vale log(1.5/1.5) = 0: hace falta un corpus mínimamente realista.
    corpus = [
        Document(page_content="Use HTTPException to return errors", metadata={"chunk_id": "a"}),
        Document(page_content="Declare a request body with Pydantic", metadata={"chunk_id": "b"}),
        Document(page_content="Add CORSMiddleware to allow origins", metadata={"chunk_id": "c"}),
        Document(page_content="Run background tasks after the response", metadata={"chunk_id": "d"}),
    ]
    bm25 = BM25Retriever.from_documents(corpus, preprocess_func=bm25_tokenize, k=1)
    assert bm25.invoke("raise an HTTPException")[0].metadata["chunk_id"] == "a"


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
