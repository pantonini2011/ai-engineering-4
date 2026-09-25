# Sistema RAG escalable en la nube con Pinecone

Pre-entrega 4 · Módulo 4 — AI Engineering (Coderhouse). La consigna original está en
[`preentrega4.md`](preentrega4.md).

Módulo de recuperación escalable para la documentación oficial del tutorial de
**FastAPI**. Hace tres cosas:

1. **Ingesta en Pinecone Serverless** (`src/ingestion.py`): lee 12 documentos Markdown, los
   corta en chunks de hasta 600 tokens, los embebe con OpenAI `text-embedding-3-small`
   (1536 dims) y los sube a Pinecone con metadatos: texto original, fuente, página, sección
   y categoría.
2. **Recuperador híbrido** (`src/rag_system.py`): la clase `RAGSystem` combina **BM25**
   (búsqueda léxica) con la **búsqueda vectorial de Pinecone** mediante un `EnsembleRetriever`
   y devuelve los 5 mejores chunks.
3. **Evaluación** (`evaluate.py`): calcula **Precision@5**, **Recall@5** (y MRR) sobre un
   golden set de 5 preguntas y compara BM25, vectorial e híbrido.

## Índice

- [Resultados](#resultados)
- [Cómo replicar el índice (paso a paso)](#cómo-replicar-el-índice-paso-a-paso)
- [Checklist de la consigna](#checklist-de-la-consigna)
- [Decisiones de diseño](#decisiones-de-diseño)
- [Esquema de cada vector en Pinecone](#esquema-de-cada-vector-en-pinecone)
- [Estructura del repo](#estructura-del-repo)
- [Tests](#tests)

## Resultados

Salida real de `python evaluate.py` (completa en
[`evidencia/04_evaluacion.txt`](evidencia/04_evaluacion.txt)):

```
Modo                       Precision@5    Recall@5     MRR
----------------------------------------------------------
BM25 (léxico)                     0.72        1.00    1.00
Vectorial (Pinecone)              0.72        1.00    1.00
Híbrido (Ensemble)                0.72        1.00    1.00
----------------------------------------------------------
Precision@5 máxima alcanzable con este corpus: 0.76 (según cuántos chunks tiene cada documento esperado).

Resumen: el recuperador híbrido encontró el documento correcto en 5/5 preguntas (Recall@5=1.00);
en promedio 3.6 de cada 5 chunks recuperados son del documento correcto (Precision@5=0.72).
```

| Pregunta (golden set) | Doc. esperado | Recall@5 | Precision@5 |
|---|---|---|---|
| Allow a frontend on another origin to call my API | `cors` | 1 | 0.60 |
| Send an email notification after returning the response | `background-tasks` | 1 | 0.20 |
| What does `include_router` do / split app into files | `bigger-applications` | 1 | 1.00 |
| Return a 404 with `HTTPException` | `handling-errors` | 1 | 0.80 |
| SQLModel `Session` dependency to save a `Hero` | `sql-databases` | 1 | 1.00 |

**Cómo leer estos números:**

- **Recall@5 = 1.00**: en las 5 preguntas, el documento correcto aparece en el top-5, y además
  siempre en la posición 1 (MRR = 1.00).
- **Precision@5 = 0.72** contra un **techo de 0.76**. La Precision@5 no puede llegar a 1.00 en
  todas las preguntas porque algunos documentos tienen menos de 5 chunks. `background-tasks`
  tiene solo 2, así que su máximo es 2/5 = 0.40 (sacó 0.20). `cors` tiene 3 (máximo 0.60, y
  sacó 0.60). `evaluate.py` calcula ese techo para no confundir un límite del corpus con un
  error del recuperador.
- **Los tres modos empatan en este golden set.** Las 5 preguntas son "fáciles" para los dos
  recuperadores: cada una tiene un tema distintivo y términos que aparecen literalmente en el
  documento. Un empate no demuestra que el híbrido sea mejor, y no ajusté las preguntas para
  que lo pareciera. Lo que sí se ve es que ambos recuperadores aportan chunks distintos. En
  [`evidencia/03_consulta_hibrida.txt`](evidencia/03_consulta_hibrida.txt), 3 de los 5
  resultados los trajeron BM25 y Pinecone a la vez, y los otros 2 solo BM25. El Ensemble
  **promueve lo que coincide en ambos rankings**, y eso lo hace más robusto que cualquiera de
  los dos por separado cuando uno falla: por ejemplo, un nombre propio que el embedding no
  distingue, o un sinónimo que BM25 no conoce. Para medir esa diferencia haría falta un golden
  set más grande y con preguntas parafraseadas.

## Cómo replicar el índice (paso a paso)

Requisitos: Python 3.12, una cuenta de [Pinecone](https://app.pinecone.io) (alcanza el plan
gratuito Starter) y una API key de OpenAI con crédito. Indexar todo el corpus consume unos
24.000 tokens de embeddings, menos de US$0,001.

```bash
# 1. Clonar e instalar
git clone https://github.com/pantonini2011/ai-engineering-4.git
cd ai-engineering-4
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell). En Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. Configurar variables: completar PINECONE_API_KEY, OPENAI_API_KEY e INDEX_NAME
copy .env.example .env            # Linux/macOS: cp .env.example .env

# 3. Crear el índice Serverless (si ya existe, solo lo valida)
python -m src.setup_index

# 4. Ingestar los documentos (idempotente; --reset vacía el namespace y re-indexa)
python -m src.ingestion

# 5. Probar el recuperador híbrido con una consulta
python -m src.rag_system "How do I add CORSMiddleware with allow_origins?"

# 6. Evaluar (Precision@5 / Recall@5)
python evaluate.py

# 7. Tests (no llaman a ninguna API)
pytest
```

> En PowerShell, si `activate` falla por la política de ejecución:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y volver a activar.

Qué hace cada paso sobre Pinecone:

| Paso | Operación en Pinecone | Qué se ve en consola |
|---|---|---|
| 3 | `list_indexes()` y, si falta, `create_index(dimension=1536, metric="cosine", spec=ServerlessSpec("aws", "us-east-1"))`. Espera a que el índice esté listo y valida dimensión y métrica. | `Índice 'fastapi-docs-rag' listo: dimension=1536, metric=cosine` |
| 4 | `upsert` por lotes de 100 en el namespace `dev`, con reintentos. Si el namespace ya tiene todos los chunks, no hace nada. | `Ingesta completa: 55 vectores subidos; namespace 'dev' ... tiene 55.` |
| 5-6 | `list` + `fetch` del namespace (para armar el corpus de BM25) y `query` vectorial. | `Corpus BM25: 55 chunks leídos de Pinecone` |

## Checklist de la consigna

Cada requisito de [`preentrega4.md`](preentrega4.md), dónde está implementado y con qué se
verifica. Los archivos de [`evidencia/`](evidencia/) son salidas reales sin editar (ver
[`evidencia/README.md`](evidencia/README.md)).

### Componentes obligatorios

| Requisito | Implementación | Evidencia |
|---|---|---|
| **Pipeline de ingesta** en Pinecone Serverless con metadatos avanzados (fuente, página, etiquetas de categoría) | [`src/ingestion.py`](src/ingestion.py) → `ingest()`: `load_documents()` → `split_documents()` → `build_embeddings()` → `upsert_chunks()` | [`01_ingesta.txt`](evidencia/01_ingesta.txt), [`06_esquema_vector.txt`](evidencia/06_esquema_vector.txt) (metadata `source`, `page`, `category`, `section`) |
| **Recuperador híbrido** (vectorial + BM25) | [`src/rag_system.py`](src/rag_system.py) → `RAGSystem` | [`03_consulta_hibrida.txt`](evidencia/03_consulta_hibrida.txt): origen `bm25+vector` por resultado |
| **Script de evaluación** con Precision@k y Recall@k sobre un golden set | [`evaluate.py`](evaluate.py) + [`data/golden_set.json`](data/golden_set.json) | [`04_evaluacion.txt`](evidencia/04_evaluacion.txt) · tests en [`tests/test_evaluate.py`](tests/test_evaluate.py) |

### Pasos sugeridos

| Paso | Implementación | Evidencia |
|---|---|---|
| Índice Serverless con dimensión 1536 para `text-embedding-3-small` | [`src/setup_index.py`](src/setup_index.py) → `ensure_index()`, con valores de [`src/config.py`](src/config.py) | Log `dimension=1536, metric=cosine` · `06_esquema_vector.txt`: `dimension del vector: 1536` |
| Texto original dentro de la metadata (sin base relacional aparte) | `upsert_chunks()`: `"metadata": {"text": c.page_content, ...}`. `RAGSystem` lee el texto de ahí tanto para BM25 (`load_corpus_from_pinecone()`) como para la búsqueda vectorial (`text_key="text"`). | `06_esquema_vector.txt` · test `test_upsert_por_lotes_con_texto_en_metadata` |
| `PineconeVectorStore` de LangChain o el SDK nativo | Los dos: SDK nativo para crear el índice, hacer upsert y leer el corpus, y `PineconeVectorStore` para la búsqueda vectorial. | — |
| `BM25Retriever` combinado con Pinecone en un `EnsembleRetriever` | `RAGSystem.__init__`: `EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5], id_key="chunk_id")` | `03_consulta_hibrida.txt` |
| JSON con pares `{"pregunta", "documento_id_esperado"}` y medir si aparecen en el top-5 | [`data/golden_set.json`](data/golden_set.json) (5 pares) · `evaluate.py` | `04_evaluacion.txt` |

### Entregable

| Requisito | Implementación | Evidencia |
|---|---|---|
| `.env` con `PINECONE_API_KEY`, `OPENAI_API_KEY` e `INDEX_NAME` | Plantilla en [`.env.example`](.env.example). El `.env` real está en `.gitignore`. `config.require()` corta con un mensaje claro si falta una variable. | — |
| Script de inicialización que verifica si el índice existe y lo crea (Serverless) | [`src/setup_index.py`](src/setup_index.py) | `01_ingesta.txt`: `El índice 'fastapi-docs-rag' ya existe: no se recrea.` · tests en [`tests/test_setup_index.py`](tests/test_setup_index.py) |
| Dataset de documentos técnicos (documentación de una librería de Python) | [`data/docs/`](data/docs/): 12 páginas del tutorial oficial de FastAPI (licencia MIT) | `01_ingesta.txt`: `12 documentos -> 55 chunks` |
| Chunks con `RecursiveCharacterTextSplitter` | `split_documents()`: `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=600, chunk_overlap=80)` con separadores de Markdown | Log `tokens por chunk: min=68, prom=433, max=596` · test `test_split_documents_respeta_tamanio_y_esquema_de_metadata` |
| Embeddings insertados con contenido y fuente en la metadata | `upsert_chunks()` | `06_esquema_vector.txt` |
| Clase `RAGSystem` que encapsula un `EnsembleRetriever` y devuelve el top-5 léxico + semántico | `RAGSystem.retrieve(query)` | `03_consulta_hibrida.txt` |
| `evaluate.py` con benchmark de 5 preguntas de fuente conocida | [`evaluate.py`](evaluate.py) | `04_evaluacion.txt` |
| Recall@5 (¿está el documento correcto entre los 5?) | `QueryResult.recall()` | Recall@5 = 1.00 |
| Precision@5 (% de los 5 que son útiles) | `QueryResult.precision()`: chunks del documento esperado / 5 | Precision@5 = 0.72 (techo 0.76) |
| Reporte breve en consola | Final de `evaluate.py`: tabla por modo y línea de "Resumen" | `04_evaluacion.txt` |
| README con los pasos para replicar el índice | [Cómo replicar el índice](#cómo-replicar-el-índice-paso-a-paso) | — |

### Errores comunes a evitar

| Error | Cómo se evita | Evidencia |
|---|---|---|
| **Mismatch de dimensiones** | La dimensión se define una sola vez (`config.EMBEDDING_DIM`) y se usa para crear el índice. `validate_index()` corta si un índice existente tiene otra dimensión, y `upsert_chunks()` verifica cada vector antes de subirlo. | Tests `test_detecta_mismatch_de_dimensiones`, `test_upsert_rechaza_dimension_incorrecta` |
| **Ignorar el namespace** | Todo upsert, query, list y fetch usa `config.NAMESPACE` (`PINECONE_NAMESPACE`, default `dev`). Además, cada vector guarda `env` en la metadata. | `06_esquema_vector.txt`: `namespaces={'dev': 55}` |
| **Subestimar el chunking** | Chunks de hasta **600 tokens** medidos con el tokenizer del propio modelo (`cl100k_base`), con 80 de solapamiento y cortes preferentemente en encabezados Markdown. | Log `min=68, prom=433, max=596`. Los chunks chicos son secciones cortas completas: no se cortan a la mitad ni se pegan con la sección siguiente. |

## Decisiones de diseño

| Decisión | Elección | Por qué |
|---|---|---|
| **Corpus** | 12 páginas del tutorial de FastAPI (path/query params, body, dependencies, security, CORS, errores, middleware, background tasks, SQL, bigger applications, first steps) | Es lo que sugiere la consigna ("documentación de una librería de Python"). Tiene muchos nombres técnicos (`HTTPException`, `APIRouter`, `CORSMiddleware`), que es justo donde BM25 aporta sobre los embeddings. |
| **Limpieza previa** | `clean_markdown()` quita las anclas `{ #slug }` y los includes `{* ../../docs_src/... *}` | Los includes apuntan a archivos de código que no están en el corpus: embebidos solo meterían ruido. |
| **Embeddings** | OpenAI `text-embedding-3-small`, 1536 dims | Lo sugerido por la consigna. Es barato y rinde bien en textos técnicos en inglés. |
| **Métrica** | Coseno | `text-embedding-3-small` devuelve vectores normalizados (norma 1). Con vectores unitarios, el coseno mide la orientación (el tema) sin importar el largo del texto, y es la métrica que recomienda OpenAI para estos embeddings. |
| **Chunking** | `RecursiveCharacterTextSplitter.from_tiktoken_encoder`, 600 tokens, 80 de solapamiento, separadores de Markdown **con `is_separator_regex=True`** | El tamaño se mide en tokens (lo que "ve" el modelo) y no en caracteres. Los separadores de Markdown de LangChain son regex (`\n#{1,6} `): sin ese flag se buscan literalmente, nunca cortan en encabezados y los chunks mezclan secciones. Lo detecté porque la metadata `section` salía vacía. |
| **Página** | `page` = posición del chunk dentro del documento (1, 2, 3…) | Markdown no tiene páginas físicas. Esto conserva el orden y permite citar "cors, página 2". |
| **IDs de vector** | Determinísticos: `<doc_id>#<nnn>` (ej. `cors#002`) | Re-ingestar el mismo documento **pisa** los vectores en vez de duplicarlos. |
| **Ingesta idempotente** | Si el namespace ya tiene tantos vectores como chunks, no re-embebe. `--reset` fuerza la re-ingesta. | No gastar embeddings en cada corrida. |
| **Resiliencia** | Upsert en lotes de 100 con hasta 4 reintentos y backoff exponencial ante `PineconeException` | Errores de red y rate limits son transitorios. Viene de la corrección del ejercicio 1 de este módulo. |
| **Corpus de BM25** | Se reconstruye desde Pinecone (`index.list()` + `fetch()`), no desde los archivos locales | BM25 necesita todo el corpus en memoria para calcular IDF. Leyéndolo del índice, BM25 y la búsqueda vectorial ven exactamente los mismos chunks, y Pinecone queda como única fuente de verdad, sin base relacional aparte. Para un corpus de millones de chunks convendría un índice léxico persistente (o sparse vectors en Pinecone), pero para esta escala alcanza. |
| **Tokenizer de BM25** | Minúsculas, `[a-z0-9_]+`, sin stopwords en inglés | El default de LangChain (`split()`) deja pegada la puntuación ("`HTTPException`," ≠ "httpexception"). Conservar `_` mantiene `add_middleware` como un solo término. Sin stopwords, BM25 trajo `sql-databases` para una pregunta de CORS solo por coincidir en "how", "do" y "with". |
| **Fusión** | `EnsembleRetriever` (Reciprocal Rank Fusion), pesos `[0.5, 0.5]`, `id_key="chunk_id"` | RRF combina rankings sin tener que normalizar scores de escalas distintas (BM25 no está acotado y el coseno va de 0 a 1). `id_key` deduplica por ID de chunk y no por texto. |
| **Precision@k** | Relevante = chunk cuyo `doc_id` es el documento esperado. Se divide por k aunque vengan menos resultados. | Es la definición de la consigna (“% de los 5 recuperados que son útiles”). Además se reporta el techo alcanzable según el tamaño de cada documento. |

## Esquema de cada vector en Pinecone

Vector real `cors#002` (completo en [`evidencia/06_esquema_vector.txt`](evidencia/06_esquema_vector.txt)):

| Campo | Ejemplo | Uso |
|---|---|---|
| `id` | `cors#002` | Determinístico: `<doc_id>#<página>` |
| `values` | 1536 floats | Embedding `text-embedding-3-small` |
| `text` | `## Use \`CORSMiddleware\` ...` | Texto original: lo devuelve la búsqueda y alimenta BM25 |
| `doc_id` | `cors` | ID del documento fuente. Es contra lo que se compara `documento_id_esperado` en la evaluación. |
| `source` | `data/docs/cors.md` | Fuente, para citar |
| `page` | `2` | Posición del chunk en el documento |
| `section` | ``Use `CORSMiddleware` `` | Encabezado Markdown al que pertenece el chunk |
| `category` | `seguridad` | Etiqueta de categoría, filtrable: `filter={"category": "seguridad"}` |
| `n_tokens` | `596` | Tamaño del chunk en tokens |
| `env` | `dev` | Entorno (coincide con el namespace) |
| `created_at` | `2026-09-25T01:38:37+00:00` | Fecha de indexación (UTC, ISO 8601) |

## Estructura del repo

```
.
├── src/
│   ├── config.py          # variables de .env, dimensión, chunking, top-k (un solo lugar)
│   ├── setup_index.py     # crea/valida el índice Serverless
│   ├── ingestion.py       # Markdown -> chunks -> embeddings -> upsert por lotes
│   └── rag_system.py      # RAGSystem: BM25 + PineconeVectorStore en un EnsembleRetriever
├── evaluate.py            # Precision@5, Recall@5, MRR por modo
├── data/
│   ├── docs/              # 12 páginas del tutorial de FastAPI (MIT)
│   └── golden_set.json    # 5 pares pregunta / documento_id_esperado
├── tests/                 # 23 tests con fakes (sin llamadas a APIs)
├── evidencia/             # salidas reales de consola
├── .env.example
└── requirements.txt       # versiones fijadas
```

## Tests

```bash
pytest -v
```

23 tests ([`evidencia/05_tests_pytest.txt`](evidencia/05_tests_pytest.txt)) que no usan red ni
API keys: reemplazan Pinecone por objetos fake. Cubren:

- limpieza de Markdown, chunking y esquema de metadata;
- upsert por lotes, reintentos y rechazo de dimensión incorrecta;
- creación y validación del índice (dimensión y métrica);
- tokenizer de BM25 y reconstrucción del corpus desde Pinecone;
- cálculo de Precision@k, Recall@k y MRR, y validación del golden set.
