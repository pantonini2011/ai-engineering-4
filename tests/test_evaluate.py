import json

import pytest
from langchain_core.documents import Document

from evaluate import QueryResult, evaluate, load_golden_set, summarize


def test_metricas_por_pregunta():
    r = QueryResult("q", "cors", ["middleware", "cors", "cors", "body", "cors"])
    assert r.recall(5) == 1.0
    assert r.precision(5) == pytest.approx(3 / 5)
    assert r.reciprocal_rank(5) == pytest.approx(1 / 2)
    assert r.recall(1) == 0.0 and r.precision(1) == 0.0


def test_precision_divide_por_k_aunque_vengan_menos_resultados():
    assert QueryResult("q", "cors", ["cors"]).precision(5) == pytest.approx(0.2)


def test_sin_acierto():
    r = QueryResult("q", "cors", ["body"] * 5)
    assert (r.recall(5), r.precision(5), r.reciprocal_rank(5)) == (0.0, 0.0, 0.0)


def test_summarize_promedia():
    res = [QueryResult("a", "x", ["x"] * 5), QueryResult("b", "y", ["z"] * 5)]
    assert summarize(res, 5) == {"precision": 0.5, "recall": 0.5, "mrr": 0.5}


def test_evaluate_usa_doc_id_de_la_metadata():
    fake = lambda q: [Document(page_content="", metadata={"doc_id": "cors"})]
    (r,) = evaluate(fake, [{"pregunta": "p", "documento_id_esperado": "cors"}])
    assert r.recuperados == ["cors"]


def test_golden_set_real_tiene_5_preguntas_con_documentos_existentes():
    from src import config

    golden = load_golden_set()
    assert len(golden) == 5
    for it in golden:
        assert (config.DOCS_DIR / f"{it['documento_id_esperado']}.md").exists()


def test_golden_set_invalido(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps([{"pregunta": "sin id"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_golden_set(p)
