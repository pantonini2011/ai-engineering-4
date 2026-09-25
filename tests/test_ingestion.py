from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from pinecone.exceptions import PineconeException

from src import config, ingestion


def test_clean_markdown_quita_anclas_e_includes():
    raw = "## Origin { #origin }\n\n{* ../../docs_src/cors/tutorial001.py hl[2] *}\n\nTexto."
    assert ingestion.clean_markdown(raw) == "## Origin\n\nTexto."


def test_section_at_devuelve_ultimo_encabezado_previo():
    text = "# A\nuno\n## B\ndos\n## C\ntres"
    assert ingestion._section_at(text, 0) == "A"
    assert ingestion._section_at(text, text.index("dos")) == "B"
    assert ingestion._section_at(text, text.index("## C")) == "C"


def test_load_documents_asigna_doc_id_fuente_y_categoria():
    docs = ingestion.load_documents()
    assert len(docs) == 12
    cors = next(d for d in docs if d.metadata["doc_id"] == "cors")
    assert cors.metadata == {"doc_id": "cors", "source": "data/docs/cors.md", "category": "seguridad"}


def test_split_documents_respeta_tamanio_y_esquema_de_metadata():
    chunks = ingestion.split_documents(ingestion.load_documents())
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "los IDs de vector deben ser únicos"
    for c in chunks:
        assert c.metadata["n_tokens"] <= config.CHUNK_SIZE_TOKENS
        assert {"doc_id", "source", "category", "chunk_id", "page", "section", "env", "created_at"} <= c.metadata.keys()
        assert c.metadata["chunk_id"] == f"{c.metadata['doc_id']}#{c.metadata['page']:03d}"


class FakeIndex:
    def __init__(self, fail_times=0):
        self.batches = []
        self.fail_times = fail_times

    def upsert(self, vectors, namespace):
        if self.fail_times:
            self.fail_times -= 1
            raise PineconeException("error transitorio")
        self.batches.append((vectors, namespace))


def _chunks(n):
    return [Document(page_content=f"t{i}", metadata={"chunk_id": f"d#{i:03d}", "doc_id": "d"}) for i in range(n)]


def test_upsert_por_lotes_con_texto_en_metadata(monkeypatch):
    monkeypatch.setattr(config, "UPSERT_BATCH_SIZE", 2)
    index = FakeIndex()
    total = ingestion.upsert_chunks(index, _chunks(5), [[0.1] * config.EMBEDDING_DIM] * 5)
    assert total == 5
    assert [len(b) for b, _ in index.batches] == [2, 2, 1]
    first = index.batches[0][0][0]
    assert first["id"] == "d#000" and first["metadata"]["text"] == "t0"
    assert all(ns == config.NAMESPACE for _, ns in index.batches)


def test_upsert_rechaza_dimension_incorrecta():
    with pytest.raises(ValueError, match="dimensión"):
        ingestion.upsert_chunks(FakeIndex(), _chunks(1), [[0.1] * 768])


def test_upsert_reintenta_errores_transitorios(monkeypatch):
    monkeypatch.setattr(ingestion.time, "sleep", lambda s: None)
    index = FakeIndex(fail_times=2)
    assert ingestion.upsert_chunks(index, _chunks(1), [[0.1] * config.EMBEDDING_DIM]) == 1
    assert len(index.batches) == 1


def test_upsert_se_rinde_tras_max_reintentos(monkeypatch):
    monkeypatch.setattr(ingestion.time, "sleep", lambda s: None)
    with pytest.raises(PineconeException):
        ingestion.upsert_chunks(FakeIndex(fail_times=99), _chunks(1), [[0.1] * config.EMBEDDING_DIM])


def test_namespace_count_lee_stats_del_namespace():
    stats = SimpleNamespace(namespaces={config.NAMESPACE: SimpleNamespace(vector_count=55)})
    index = SimpleNamespace(describe_index_stats=lambda: stats)
    assert ingestion.namespace_count(index) == 55
    empty = SimpleNamespace(describe_index_stats=lambda: SimpleNamespace(namespaces={}))
    assert ingestion.namespace_count(empty) == 0
