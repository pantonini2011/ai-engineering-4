from types import SimpleNamespace

import pytest

from src import config, setup_index


def _desc(dimension=config.EMBEDDING_DIM, metric=config.METRIC):
    return SimpleNamespace(
        name=config.INDEX_NAME, dimension=dimension, metric=metric,
        host="h", status={"ready": True},
    )


class FakePinecone:
    def __init__(self, existing, desc=None):
        self.existing = list(existing)
        self.desc = desc or _desc()
        self.created = []

    def list_indexes(self):
        return SimpleNamespace(names=lambda: self.existing)

    def create_index(self, **kwargs):
        self.created.append(kwargs)
        self.existing.append(kwargs["name"])

    def describe_index(self, name):
        return self.desc

    def Index(self, name):
        return f"index:{name}"


def test_crea_indice_serverless_si_no_existe():
    pc = FakePinecone(existing=[])
    assert setup_index.ensure_index(pc) == f"index:{config.INDEX_NAME}"
    (kwargs,) = pc.created
    assert kwargs["dimension"] == 1536 and kwargs["metric"] == "cosine"
    assert kwargs["spec"].cloud == config.PINECONE_CLOUD


def test_no_recrea_indice_existente():
    pc = FakePinecone(existing=[config.INDEX_NAME])
    setup_index.ensure_index(pc)
    assert pc.created == []


def test_detecta_mismatch_de_dimensiones():
    pc = FakePinecone(existing=[config.INDEX_NAME], desc=_desc(dimension=768))
    with pytest.raises(ValueError, match="Mismatch de dimensiones"):
        setup_index.ensure_index(pc)


def test_detecta_metrica_distinta():
    with pytest.raises(ValueError, match="metric"):
        setup_index.validate_index(_desc(metric="euclidean"))
