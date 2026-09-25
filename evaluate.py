"""Evaluación del recuperador con un Golden Set: Precision@k, Recall@k y MRR.

Uso:  python evaluate.py            (k=5 por default)
      python evaluate.py --k 3

Cada ítem del golden set (data/golden_set.json) es
    {"pregunta": "...", "documento_id_esperado": "<doc_id>"}
y un chunk recuperado se considera relevante si su metadata.doc_id coincide.

- Recall@k    ¿el documento correcto aparece entre los k recuperados? (1 o 0 por
              pregunta; hay un solo documento relevante por pregunta)
- Precision@k fracción de los k recuperados que son del documento correcto
- MRR         1 / posición del primer chunk relevante (extra: premia que el
              documento correcto aparezca arriba, no solo que aparezca)
- Hit Rate    fracción de preguntas con al menos un acierto en el top-k. Con un
              solo documento relevante por pregunta coincide con Recall@k.

Se evalúan los tres modos (BM25 solo, vectorial solo, híbrido) sobre la misma
instancia de RAGSystem para mostrar qué aporta la combinación.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from src import config


@dataclass
class QueryResult:
    pregunta: str
    esperado: str
    recuperados: list[str]  # doc_id de cada chunk, en orden de ranking

    def precision(self, k: int) -> float:
        return sum(d == self.esperado for d in self.recuperados[:k]) / k

    def recall(self, k: int) -> float:
        return float(self.esperado in self.recuperados[:k])

    def reciprocal_rank(self, k: int) -> float:
        for pos, d in enumerate(self.recuperados[:k], start=1):
            if d == self.esperado:
                return 1 / pos
        return 0.0


def load_golden_set(path: Path = config.GOLDEN_SET_PATH) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    for it in items:
        if not {"pregunta", "documento_id_esperado"} <= it.keys():
            raise ValueError(f"Ítem inválido en {path}: {it}")
    return items


def evaluate(retrieve, golden: list[dict]) -> list[QueryResult]:
    return [
        QueryResult(
            pregunta=it["pregunta"],
            esperado=it["documento_id_esperado"],
            recuperados=[d.metadata["doc_id"] for d in retrieve(it["pregunta"])],
        )
        for it in golden
    ]


def summarize(results: list[QueryResult], k: int) -> dict[str, float]:
    n = len(results)
    return {
        "precision": sum(r.precision(k) for r in results) / n,
        "recall": sum(r.recall(k) for r in results) / n,
        "mrr": sum(r.reciprocal_rank(k) for r in results) / n,
    }


def print_detail(results: list[QueryResult], k: int) -> None:
    for i, r in enumerate(results, start=1):
        marcas = " ".join("✔" if d == r.esperado else "·" for d in r.recuperados[:k])
        print(f"\n[{i}] {r.pregunta}")
        print(f"    esperado: {r.esperado}")
        print(f"    top-{k}:   {r.recuperados[:k]}")
        print(
            f"    relevantes: {marcas}  ->  Hit: {'SÍ' if r.recall(k) else 'NO'}  Recall@{k}={r.recall(k):.0f}  "
            f"Precision@{k}={r.precision(k):.2f}  RR={r.reciprocal_rank(k):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el recuperador con el golden set.")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    args = parser.parse_args()
    k = args.k

    config.setup_logging()
    from src.rag_system import RAGSystem  # import diferido: el cálculo de métricas no necesita APIs

    golden = load_golden_set()
    rag = RAGSystem(k=k)

    # Cuántos chunks tiene cada documento esperado: acota la Precision@k máxima
    # posible (un doc con 2 chunks nunca puede llenar un top-5).
    chunks_por_doc: dict[str, int] = {}
    for d in rag.bm25_retriever.docs:
        chunks_por_doc[d.metadata["doc_id"]] = chunks_por_doc.get(d.metadata["doc_id"], 0) + 1
    precision_max = sum(min(chunks_por_doc.get(it["documento_id_esperado"], 0), k) / k for it in golden) / len(golden)

    modos = {
        "BM25 (léxico)": rag.retrieve_bm25,
        "Vectorial (Pinecone)": rag.retrieve_vector,
        "Híbrido (Ensemble)": rag.retrieve,
    }
    resumen = {}
    for nombre, fn in modos.items():
        resultados = evaluate(fn, golden)
        resumen[nombre] = summarize(resultados, k)
        if nombre.startswith("Híbrido"):
            detalle = resultados

    print("\n" + "=" * 78)
    print(f"EVALUACIÓN DEL RECUPERADOR — índice '{config.INDEX_NAME}', namespace '{rag.namespace}'")
    print(f"Golden set: {len(golden)} preguntas · k={k} · pesos Ensemble [BM25, vector]={config.ENSEMBLE_WEIGHTS}")
    print("=" * 78)
    print("\nDetalle por pregunta (modo híbrido):")
    print_detail(detalle, k)

    print(f"\n{'Modo':<24}{'Precision@'+str(k):>14}{'Recall@'+str(k):>12}{'MRR':>8}")
    print("-" * 58)
    for nombre, m in resumen.items():
        print(f"{nombre:<24}{m['precision']:>14.2f}{m['recall']:>12.2f}{m['mrr']:>8.2f}")
    print("-" * 58)
    print(
        f"Precision@{k} máxima alcanzable con este corpus: {precision_max:.2f} "
        f"(según cuántos chunks tiene cada documento esperado)."
    )
    h = resumen["Híbrido (Ensemble)"]
    n = len(golden)
    hits = sum(r.recall(k) for r in detalle)
    print(f"\nMÉTRICAS GLOBALES DEL HÍBRIDO SOBRE {n} PREGUNTAS (namespace '{rag.namespace}'):")
    print(f"  • Recall@{k} promedio:    {h['recall']:.2f} ({h['recall']:.0%})")
    print(f"  • Precision@{k} promedio: {h['precision']:.2f} ({h['precision']:.0%})")
    print(f"  • Hit Rate:             {hits / n:.2f} ({hits:.0f}/{n})")
    print(f"  • MRR:                  {h['mrr']:.2f}")
    print(
        f"\nResumen: el recuperador híbrido encontró el documento correcto en "
        f"{h['recall'] * n:.0f}/{n} preguntas (Recall@{k}={h['recall']:.2f}); "
        f"en promedio {h['precision'] * k:.1f} de cada {k} chunks recuperados son del documento "
        f"correcto (Precision@{k}={h['precision']:.2f})."
    )


if __name__ == "__main__":
    main()
