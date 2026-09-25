# Evidencia de ejecución

Salidas reales de consola, sin editar, capturadas con `tee` el 24/09/2026
contra el índice Serverless `fastapi-docs-rag` (aws/us-east-1), namespace `dev`.

| Archivo | Comando | Qué demuestra |
|---|---|---|
| [`01_ingesta.txt`](01_ingesta.txt) | `python -m src.ingestion` | El índice ya existía (lo creó `python -m src.setup_index`) y se valida su dimensión 1536 y métrica coseno. 12 documentos → 55 chunks de 68 a 596 tokens, 55 embeddings con `text-embedding-3-small` y upsert por lotes. |
| [`02_ingesta_idempotente.txt`](02_ingesta_idempotente.txt) | `python -m src.ingestion` (2ª vez) | Si el namespace ya tiene los 55 vectores, no re-indexa ni gasta embeddings. |
| [`03_consulta_hibrida.txt`](03_consulta_hibrida.txt) | `python -m src.rag_system "How do I add CORSMiddleware with allow_origins?"` | `RAGSystem` devuelve el top-5 e indica qué recuperador trajo cada chunk (`bm25`, `vector` o ambos). |
| [`04_evaluacion.txt`](04_evaluacion.txt) | `python evaluate.py` | Precision@5, Recall@5 y MRR por pregunta y por modo (BM25, vectorial, híbrido). |
| [`05_tests_pytest.txt`](05_tests_pytest.txt) | `pytest -v` | 23 tests que no llaman a ninguna API (usan fakes). |
| [`06_esquema_vector.txt`](06_esquema_vector.txt) | fetch de `cors#002` (comando abajo) | Un vector tal como quedó guardado: 1536 dims, el texto original en `metadata.text`, y fuente, página, sección y categoría. `describe_index_stats` muestra 55 vectores en el namespace `dev`. |

El comando de `06_esquema_vector.txt` no es un script del repo; es un fetch puntual:

```bash
python -c "from src import config; from src.setup_index import get_client; \
idx = get_client().Index(config.INDEX_NAME); \
print(idx.fetch(ids=['cors#002'], namespace=config.NAMESPACE).vectors['cors#002'].metadata)"
```

Nota: Pinecone guarda los números de la metadata como float (`"page": 2.0`). Los filtros
numéricos (`{"page": {"$lte": 2}}`) funcionan igual.
