# Sistema RAG escalable en la nube con Pinecone

Pre-entrega 4 · Módulo 4 — AI Engineering (Coderhouse). La consigna original está en
[`preentrega4.md`](preentrega4.md).

## Índice

1. [Descripción del proyecto](#1-descripción-del-proyecto)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Decisiones de diseño](#3-decisiones-de-diseño)
4. [Esquema de cada vector en Pinecone](#4-esquema-de-cada-vector-en-pinecone)
5. [Cómo replicar el índice (paso a paso)](#5-cómo-replicar-el-índice-paso-a-paso)
6. [Resultados y evaluación](#6-resultados-y-evaluación)
7. [Tests y verificaciones](#7-tests-y-verificaciones)
8. [Checklist de la consigna](#8-checklist-de-la-consigna)

## 1. Descripción del proyecto

Módulo de recuperación escalable para la documentación oficial del tutorial de **FastAPI en
español** (12 páginas: parámetros, body, dependencias, seguridad, CORS, errores, middleware,
tareas en segundo plano, SQL, etc.). Un pipeline de ingesta corta los documentos en chunks de
hasta 600 tokens, los embebe con OpenAI `text-embedding-3-small` (1536 dimensiones) y los sube
a un índice **Pinecone Serverless** (AWS us-east-1, métrica coseno), en un namespace por
entorno. Cada vector guarda en su metadata el texto original, la fuente, la página, la sección,
la categoría y la fecha de indexación, así que no hace falta otra base de datos para
recuperar el contenido.

La recuperación la hace la clase `RAGSystem`, que encapsula un **`EnsembleRetriever`** de
LangChain. Combina **BM25** (búsqueda léxica, fuerte con nombres técnicos exactos como
`HTTPException`) con la **búsqueda vectorial de Pinecone** (semántica, fuerte con paráfrasis)
y devuelve los 5 mejores chunks. El script `evaluate.py` mide **Recall@5** y **Precision@5**
sobre un golden set de 5 preguntas en español y compara BM25, vectorial e híbrido. Resultado:
el documento correcto aparece en el top-5 en las 5 preguntas (Recall@5 = 1.00), con
Precision@5 = 0.76 para el híbrido.

## 2. Estructura del repositorio

```
ai-engineering-4/
├── src/
│   ├── config.py            # variables de .env y parámetros (dimensión, chunking, top-k, pesos)
│   ├── setup_index.py       # verifica si el índice existe, lo crea (Serverless) y lo valida
│   ├── ingestion.py         # Markdown -> chunks -> embeddings -> upsert por lotes con reintentos
│   └── rag_system.py        # clase RAGSystem: BM25 + PineconeVectorStore en un EnsembleRetriever
├── evaluate.py              # Precision@5, Recall@5 y MRR por modo (BM25 / vectorial / híbrido)
├── data/
│   ├── docs/                # 12 páginas del tutorial de FastAPI en español (licencia MIT)
│   └── golden_set.json      # 5 pares {"pregunta", "documento_id_esperado"}
├── tests/                   # 47 tests con fakes (no llaman a ninguna API)
│   ├── test_setup_index.py
│   ├── test_ingestion.py
│   ├── test_rag_system.py
│   └── test_evaluate.py
├── evidencia/               # salidas reales de consola de cada paso (ver evidencia/README.md)
├── .env.example             # plantilla de variables (PINECONE_API_KEY, OPENAI_API_KEY, INDEX_NAME)
├── requirements.txt         # dependencias con versiones fijadas
├── pytest.ini
└── preentrega4.md           # consigna
```

## 3. Decisiones de diseño

### 3.1 Modelo de embeddings y dimensión

| Parámetro | Valor | Dónde |
|---|---|---|
| Modelo | OpenAI `text-embedding-3-small` | `config.EMBEDDING_MODEL` |
| Dimensión | **1536** | `config.EMBEDDING_DIM` |
| Tokenizer para medir chunks | `cl100k_base` (el del propio modelo) | `ingestion.split_documents()` |

- **Por qué este modelo:** es el que sugiere la consigna, es barato (indexar todo el corpus
  consume unos 28.000 tokens, menos de US$0,001) y es **multilingüe**. Indexa y consulta en
  español sin cambiar nada, y entiende los términos técnicos en inglés que la documentación
  deja sin traducir ("request", "response", `APIRouter`).
- **Por qué 1536:** es la dimensión de salida nativa del modelo. El índice se crea con esa
  dimensión, y el valor está definido **en un solo lugar** (`config.py`) para evitar el
  *mismatch* de dimensiones que advierte la consigna. Hay dos controles adicionales:
  `setup_index.validate_index()` corta si un índice existente tiene otra dimensión, y
  `ingestion.upsert_chunks()` verifica el largo de cada vector antes de subirlo.
- **El mismo modelo para indexar y consultar:** `build_embeddings()` es la única función que
  crea el modelo de embeddings, y la usan tanto la ingesta como `RAGSystem`.

### 3.2 Métrica de similitud: coseno

El índice se crea con `metric="cosine"`. Justificación técnica:

- **Vectores normalizados:** `text-embedding-3-small` devuelve vectores de norma L2 = 1. Para
  vectores unitarios, `cos(a, b) = a·b / (‖a‖‖b‖) = a·b`, así que el coseno mide solamente la
  **orientación** del vector, es decir, el tema del texto.
- **Independencia de la magnitud:** en embeddings de texto, la magnitud puede variar con el
  largo o la densidad del texto, y no con su significado. El coseno ignora la magnitud: un
  chunk de 100 tokens y uno de 590 sobre el mismo tema quedan igual de cerca de la pregunta.
  Con distancia euclidiana, los textos más largos o más cortos quedarían penalizados sin
  razón semántica.
- **Recomendación del proveedor:** OpenAI recomienda la similitud coseno para sus embeddings.
  Con vectores normalizados, `dotproduct` daría el mismo ranking, pero `cosine` es más robusto
  si algún día se cambia a un modelo que no normalice.
- **Validación:** `setup_index.validate_index()` también verifica que un índice existente
  tenga métrica `cosine`, y si no la tiene, corta.

### 3.3 Estrategia de recuperación híbrida (`EnsembleRetriever`)

```python
EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],   # léxico, semántico
    weights=[0.5, 0.5],                              # config.ENSEMBLE_WEIGHTS
    id_key="chunk_id",
)
```

- **Dos recuperadores complementarios.** BM25 (`BM25Retriever`) matchea palabras exactas y
  es fiable con nombres técnicos (`CORSMiddleware`, `include_router`) que el embedding puede
  "diluir". La búsqueda vectorial (`PineconeVectorStore`, top-5 en el namespace) entiende
  paráfrasis y sinónimos: "que el cliente no espere" se relaciona con "tareas en segundo
  plano" aunque no compartan palabras.
- **Fusión con Reciprocal Rank Fusion (RRF).** Cada chunk suma `peso / (60 + posición)` por
  cada ranking en el que aparece. RRF combina **posiciones** y no scores, así que no hace falta
  normalizar escalas incompatibles: BM25 no está acotado y el coseno va de 0 a 1. Un chunk que
  aparece en los dos rankings sube al tope.
- **Pesos `[0.5, 0.5]` (léxico / semántico).** Empecé con pesos iguales, sin preferir ninguna
  señal. En la evaluación el vectorial solo rinde algo más que el híbrido (ver
  [Resultados](#6-resultados-y-evaluación)). Aun así, **no ajusté los pesos mirando esas
  mismas 5 preguntas**: con un golden set tan chico eso sería sobreajustar la evaluación.
  Calibrarlos requiere un conjunto de validación separado.
- **`id_key="chunk_id"`.** Deduplica por el ID del chunk y no por el texto: si ambos
  recuperadores traen el mismo chunk, cuenta una sola vez y suma los dos aportes.
- **El corpus de BM25 sale de Pinecone.** BM25 necesita todo el corpus en memoria para
  calcular IDF. `load_corpus_from_pinecone()` lo reconstruye con `index.list()` + `fetch()`
  leyendo `metadata.text`, así BM25 y la búsqueda vectorial ven exactamente los mismos chunks y
  Pinecone es la única fuente de verdad. Para millones de chunks convendría un índice léxico
  persistente (o sparse vectors en Pinecone), pero a esta escala alcanza.
- **Tokenizer de BM25 para español** (`bm25_tokenize()`): pasa todo a minúsculas, quita las
  tildes ("configuración" = "configuracion"), conserva `_` (`add_middleware` queda como un
  solo término) y filtra stopwords en español (sin eso, BM25 trae chunks de otros temas solo
  por coincidir en "cómo", "de" o "que").
- **Umbral de relevancia para preguntas fuera de tema.** Un recuperador top-k siempre devuelve
  k chunks, aunque la pregunta no tenga nada que ver con el corpus: "¿Cuándo me puedo tomar
  vacaciones?" traía 5 chunks de FastAPI. Para evitarlo hay dos controles:
  - Si la similitud coseno del mejor chunk vectorial queda por debajo de `MIN_SIMILITUD`
    (0.38 por default), el sistema devuelve "sin resultados relevantes" en vez de un top-5.
    El valor sale de medir las dos poblaciones en este corpus: las preguntas del golden set
    dan entre 0.45 y 0.63 y las preguntas fuera de tema entre 0.12 y 0.31 (vacaciones 0.21,
    Nginx 0.31). Con tan pocas preguntas es un valor orientativo: se puede ajustar en `.env`
    sin tocar código.
  - BM25 descarta los chunks con score 0, es decir, sin ningún término en común con la
    consulta. El `BM25Retriever` de LangChain los devolvía igual para completar el top-k,
    como relleno.

  La evidencia está en
  [`evidencia/09_pregunta_fuera_de_dominio.txt`](evidencia/09_pregunta_fuera_de_dominio.txt).

### 3.4 Estrategia de segmentación: namespaces y metadata

**Namespaces por entorno.** Todas las operaciones (upsert, query, list, fetch, delete) usan
`config.NAMESPACE`, que viene de la variable `PINECONE_NAMESPACE` (default `dev`):

- Una re-ingesta de prueba en `dev` (por ejemplo `python -m src.ingestion --reset`) no toca
  los vectores de `prod`. Para indexar en producción alcanza con `PINECONE_NAMESPACE=prod`.
- La búsqueda se limita al namespace, así que nunca se mezclan datos de distintos entornos o
  tenants y la consulta no recorre vectores ajenos.
- El conteo por namespace se verifica con `describe_index_stats()`
  ([`evidencia/06_esquema_vector.txt`](evidencia/06_esquema_vector.txt):
  `namespaces={'dev': 65}`).

**Metadata para filtrar y citar.** Además del texto, cada vector lleva campos que permiten
filtrar dentro del namespace (ver el esquema completo en la [sección 4](#4-esquema-de-cada-vector-en-pinecone)):

| Campo | Filtro de ejemplo | Para qué |
|---|---|---|
| `category` | `{"category": "seguridad"}` | Buscar solo en un área temática (seguridad, errores, persistencia…) |
| `doc_id` / `source` | `{"doc_id": "cors"}` | Limitar a un documento y citar la fuente |
| `page` | `{"page": {"$lte": 2}}` | Quedarse con el inicio de cada documento |
| `env` | `{"env": "prod"}` | Control extra si alguna vez se mezclan entornos |
| `created_at` | — | Detectar si el índice está desactualizado respecto de `data/docs/` |

**Filtrado en la recuperación híbrida.** `RAGSystem.retrieve(query, filtro=...)` recibe un
filtro con la sintaxis de metadata de Pinecone (`{"campo": valor}`, `$eq`, `$ne`, `$gt`,
`$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$and`, `$or`) y lo aplica a **los dos**
recuperadores:

- **Vectorial:** el filtro va a Pinecone (`similarity_search(..., filter=filtro)`), que lo
  aplica del lado del servidor dentro del namespace.
- **BM25:** corre en memoria, así que `cumple_filtro()` evalúa el mismo filtro sobre la
  metadata de cada chunk.
- **Fusión:** RRF con los mismos pesos (`EnsembleRetriever.weighted_reciprocal_rank()`).

En los dos casos el filtro se aplica **antes** de cortar el top-5: el resultado sale completo
del subconjunto filtrado y no de un top-5 general al que después se le sacan elementos (que
podría quedar vacío).

```python
rag = RAGSystem()
rag.retrieve("¿Cómo agrego headers personalizados?", filtro={"category": "seguridad"})
rag.retrieve("...", filtro={"doc_id": {"$in": ["handling-errors", "middleware"]}, "page": {"$lte": 2}})
```

Ejemplo real ([`evidencia/07_consulta_con_filtros.txt`](evidencia/07_consulta_con_filtros.txt)):
sin filtro, "¿Cómo agrego headers personalizados a la respuesta?" mezcla chunks de `errores`,
`infraestructura` y `seguridad`. Con `--categoria seguridad`, los 5 resultados son de
seguridad. Con el filtro `$in` + `page $lte 2` salen solo 4, porque son los únicos chunks
que lo cumplen. La evaluación no usa filtros: mide la recuperación sobre todo el namespace.

### 3.5 Otras decisiones

| Decisión | Elección | Por qué |
|---|---|---|
| **Corpus** | Traducción oficial al español del tutorial de FastAPI | La consigna sugiere documentación de una librería de Python. Tiene muchos nombres técnicos, justo donde BM25 aporta. |
| **Limpieza previa** | `clean_markdown()` quita las anclas `{ #slug }` y los includes `{* ../../docs_src/... *}` | Los includes apuntan a archivos de código que no están en el corpus: embebidos solo meterían ruido. |
| **Chunking** | `RecursiveCharacterTextSplitter.from_tiktoken_encoder`, **600 tokens**, 80 de solapamiento, separadores de Markdown con `is_separator_regex=True` | Dentro del rango de 500-800 tokens que sugiere la consigna, medido en tokens y no en caracteres. Los separadores de Markdown son regex (`\n#{1,6} `): sin ese flag nunca cortan en encabezados y los chunks mezclan secciones. Resultado: 65 chunks de 33 a 592 tokens (promedio 427). Los más chicos son secciones cortas completas, como "Resumen". |
| **Página** | `page` = posición del chunk dentro del documento | Markdown no tiene páginas físicas. Esto conserva el orden y permite citar "cors, página 2". |
| **IDs de vector** | Determinísticos: `<doc_id>#<nnn>` (ej. `cors#002`) | Re-ingestar pisa los vectores en vez de duplicarlos. |
| **Ingesta idempotente** | Si el namespace ya tiene tantos vectores como chunks, no re-embebe. `--reset` fuerza la re-ingesta. | No gastar embeddings en cada corrida. |
| **Resiliencia** | Upsert en lotes de 100 con hasta 4 reintentos y backoff exponencial ante `PineconeException` | Los errores de red y los rate limits son transitorios. |
| **Precision@k** | Relevante = chunk cuyo `doc_id` es el documento esperado. Se divide por k aunque vengan menos resultados. | Es la definición de la consigna ("% de los 5 recuperados que son útiles"). |

## 4. Esquema de cada vector en Pinecone

Payload de cada registro en el `upsert` (`ingestion.upsert_chunks()`), con los valores reales
del vector `cors#002` ([`evidencia/06_esquema_vector.txt`](evidencia/06_esquema_vector.txt)):

```json
{
  "id": "cors#002",
  "values": [-0.0034, 0.0102, 0.0486, "...", -0.0203],
  "metadata": {
    "text": "## Usa `CORSMiddleware`\n\nPuedes configurarlo en tu aplicación **FastAPI** usando el `CORSMiddleware`...",
    "doc_id": "cors",
    "chunk_id": "cors#002",
    "source": "data/docs/cors.md",
    "page": 2,
    "section": "Usa `CORSMiddleware`",
    "category": "seguridad",
    "n_tokens": 560,
    "env": "dev",
    "created_at": "2026-09-25T02:46:21+00:00"
  }
}
```

(`values` está abreviado: son 1536 floats; se muestran los 3 primeros y el último.)

| Campo | Tipo | Uso |
|---|---|---|
| `id` | string | Determinístico: `<doc_id>#<página>`. Re-ingestar pisa, no duplica. |
| `values` | float[1536] | Embedding `text-embedding-3-small` |
| `metadata.text` | string | **Texto original** del chunk. Lo devuelve la búsqueda vectorial (`text_key="text"`) y alimenta BM25: no hace falta una base relacional aparte. |
| `metadata.doc_id` | string | ID del documento fuente. Es lo que se compara con `documento_id_esperado` en la evaluación. |
| `metadata.chunk_id` | string | Igual al `id`. Lo usa `EnsembleRetriever` (`id_key`) para deduplicar. |
| `metadata.source` | string | Ruta del archivo fuente, para citar |
| `metadata.page` | number | Posición del chunk en el documento |
| `metadata.section` | string | Encabezado Markdown al que pertenece el chunk |
| `metadata.category` | string | Etiqueta de categoría (seguridad, errores, persistencia…), filtrable |
| `metadata.n_tokens` | number | Tamaño del chunk en tokens |
| `metadata.env` | string | Entorno; coincide con el namespace |
| `metadata.created_at` | string | Fecha de indexación (UTC, ISO 8601) |

> Pinecone guarda los números de la metadata como float (`"page": 2.0`). Los filtros
> numéricos funcionan igual.

## 5. Cómo replicar el índice (paso a paso)

Requisitos: Python 3.12, una cuenta de [Pinecone](https://app.pinecone.io) (alcanza el plan
gratuito Starter) y una API key de OpenAI con crédito cargado.

### 5.1 Clonación y entorno virtual

```bash
git clone https://github.com/pantonini2011/ai-engineering-4.git
cd ai-engineering-4
python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell)
# source .venv/bin/activate       # Linux / macOS
```

> En PowerShell, si `activate` falla por la política de ejecución:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y volver a activar.

### 5.2 Instalación de dependencias

```bash
pip install -r requirements.txt
```

### 5.3 Configuración de `.env`

```bash
copy .env.example .env            # Windows
# cp .env.example .env            # Linux / macOS
```

Completar en `.env`:

| Variable | Obligatoria | Valor |
|---|---|---|
| `PINECONE_API_KEY` | Sí | API key de Pinecone |
| `OPENAI_API_KEY` | Sí | API key de OpenAI (para los embeddings) |
| `INDEX_NAME` | Sí | Nombre del índice, ej. `fastapi-docs-rag` |
| `PINECONE_NAMESPACE` | No (default `dev`) | Entorno: `dev` / `prod` |
| `PINECONE_CLOUD`, `PINECONE_REGION` | No (default `aws`, `us-east-1`) | Ubicación del índice Serverless |
| `MIN_SIMILITUD` | No (default `0.38`) | Similitud coseno mínima para considerar que la pregunta es del dominio ([sección 3.3](#33-estrategia-de-recuperación-híbrida-ensembleretriever)) |

Si falta una variable obligatoria, los scripts cortan con un mensaje que dice cuál falta.

### 5.4 Comandos de ejecución

```bash
# Inicialización: crea el índice Serverless si no existe (si existe, valida dimensión y métrica)
python -m src.setup_index

# Ingesta: chunking + embeddings + upsert en el namespace
python -m src.ingestion            # idempotente: si ya está poblado, no hace nada
python -m src.ingestion --reset    # vacía el namespace y re-indexa todo

# Recuperación: top-5 híbrido para una consulta
python -m src.rag_system "¿Cómo agrego CORSMiddleware con allow_origins?"
python -m src.rag_system "¿Cómo agrego CORSMiddleware con allow_origins?" --json   # misma consulta, salida en JSON

# Recuperación con filtro de metadata (por categoría, o cualquier filtro de Pinecone en JSON)
python -m src.rag_system "¿Cómo agrego headers personalizados a la respuesta?" --categoria seguridad
python -m src.rag_system "¿Cómo agrego headers?" --filtro '{"page": {"$lte": 1}}'        # bash / Linux / macOS
python -m src.rag_system "¿Cómo agrego headers?" --filtro '{\"page\": {\"$lte\": 1}}'    # PowerShell 5.1

# Evaluación: Precision@5 y Recall@5 sobre el golden set (evaluate.py está en la raíz, no en src/)
python evaluate.py
```

Qué hace cada comando sobre Pinecone:

| Comando | Operación en Pinecone | Qué se ve en consola |
|---|---|---|
| `src.setup_index` | `list_indexes()` y, si falta, `create_index(dimension=1536, metric="cosine", spec=ServerlessSpec("aws", "us-east-1"))`. Espera a que esté listo y valida dimensión y métrica. | `Índice 'fastapi-docs-rag' listo: dimension=1536, metric=cosine` |
| `src.ingestion` | `ensure_index()` (mismo chequeo) + `upsert` por lotes de 100 en el namespace, con reintentos | `12 documentos -> 65 chunks` · `Ingesta completa: 65 vectores subidos; namespace 'dev' ... tiene 65.` |
| `src.rag_system` | `list` + `fetch` del namespace (corpus de BM25) y `query` vectorial (con `filter` si se pasa `--categoria` o `--filtro`) | Índice, namespace y estrategia; top-5 con `chunk_id`, score combinado (RRF), categoría, fuente, `page`, sección, un extracto y qué recuperador lo trajo (`bm25`, `vector` o ambos). Con `--json`, lo mismo como JSON |
| `evaluate.py` | Igual que `src.rag_system`, para las 5 preguntas y los 3 modos | Detalle por pregunta (con Hit SÍ/NO), tabla de métricas por modo y métricas globales (Recall@5, Precision@5, Hit Rate, MRR) |

Salida de la consulta ([`evidencia/03_consulta_hibrida.txt`](evidencia/03_consulta_hibrida.txt);
la versión `--json` está en [`evidencia/08_consulta_json.txt`](evidencia/08_consulta_json.txt)):

```
Consulta: ¿Cómo agrego CORSMiddleware con allow_origins?
Índice: fastapi-docs-rag · namespace: dev · estrategia: híbrida (BM25 + Pinecone, RRF, pesos [0.5, 0.5])
Top-5 (similitud coseno máxima 0.749, umbral 0.38):
  1. cors#002                 score=0.0164  [seguridad]  data/docs/cors.md · page 2  (de: bm25+vector)
     'Usa `CORSMiddleware`': ## Usa `CORSMiddleware` Puedes configurarlo en tu aplicación **FastAPI** usando el `CORSMi...
  2. cors#003                 score=0.0160  [seguridad]  data/docs/cors.md · page 3  (de: bm25+vector)
     'Usa `CORSMiddleware`': Ninguno de `allow_origins`, `allow_methods` y `allow_headers` puede establecerse a `['*']`...
  3. cors#004                 score=0.0156  [seguridad]  data/docs/cors.md · page 4  (de: bm25+vector)
     'Requests de preflight CORS': ### Requests de preflight CORS Estos son cualquier request `OPTIONS` con headers `Origin`...
  4. middleware#001           score=0.0154  [infraestructura]  data/docs/middleware.md · page 1  (de: bm25+vector)
     'Middleware': # Middleware Puedes añadir middleware a las aplicaciones de **FastAPI**. Un "middleware" e...
  5. cors#001                 score=0.0081  [seguridad]  data/docs/cors.md · page 1  (de: vector)
     'CORS (Cross-Origin Resource Sharing)': # CORS (Cross-Origin Resource Sharing) [CORS o "Cross-Origin Resource Sharing"](https://de...
```

El `score` es el score RRF combinado (ver [sección 3.3](#33-estrategia-de-recuperación-híbrida-ensembleretriever)). Con pesos 0.5 el máximo es
0.5/61 + 0.5/61 ≈ 0.0164, que corresponde a un chunk que sale primero en los dos rankings. Un
chunk que trae un solo recuperador no pasa de 0.5/61 ≈ 0.0082.

La misma consulta con `--json` (se muestra el primero de los 5 fragmentos; la salida completa
está en [`evidencia/08_consulta_json.txt`](evidencia/08_consulta_json.txt)):

```json
{
  "pregunta": "¿Cómo agrego CORSMiddleware con allow_origins?",
  "estrategia_recuperacion": "hibrida_ensemble (bm25 + pinecone_dense, RRF)",
  "pesos": {"bm25": 0.5, "vector": 0.5},
  "indice": "fastapi-docs-rag",
  "namespace": "dev",
  "filtro": null,
  "top_k": 5,
  "similitud_maxima": 0.749,
  "umbral_similitud": 0.38,
  "fragmentos_recuperados": [
    {
      "chunk_id": "cors#002",
      "doc_id": "cors",
      "fuente": "data/docs/cors.md",
      "categoria": "seguridad",
      "seccion": "Usa `CORSMiddleware`",
      "page": 2,
      "score_combinado": 0.0164,
      "similitud_coseno": 0.749,
      "recuperado_por": ["bm25", "vector"],
      "extracto": "## Usa `CORSMiddleware` Puedes configurarlo en tu aplicación **FastAPI** usando el `CORSMiddleware`. * Importa `CORSMiddleware`. * Crea una lista de orígenes permitidos (como strings). * Agrégalo como..."
    }
  ],
  "fuentes": ["data/docs/cors.md", "data/docs/middleware.md"]
}
```

Una pregunta fuera de la documentación no devuelve chunks: la similitud coseno máxima queda
por debajo de `MIN_SIMILITUD` (ver [sección 3.3](#33-estrategia-de-recuperación-híbrida-ensembleretriever);
salida completa en [`evidencia/09_pregunta_fuera_de_dominio.txt`](evidencia/09_pregunta_fuera_de_dominio.txt)):

```
Consulta: ¿Cuando me puedo tomar vacaciones?
Índice: fastapi-docs-rag · namespace: dev · estrategia: híbrida (BM25 + Pinecone, RRF, pesos [0.5, 0.5])
Sin resultados relevantes: la similitud máxima (0.210) está por debajo del umbral (0.38). La pregunta parece estar fuera de la documentación indexada.
```

Con `--json`, `fragmentos_recuperados` y `fuentes` vienen vacíos, y el campo `mensaje` explica el
motivo.

La salida real de cada comando está en [`evidencia/`](evidencia/).

## 6. Resultados y evaluación

### 6.1 Benchmark: golden set de 5 preguntas

[`data/golden_set.json`](data/golden_set.json): cada pregunta tiene un documento fuente
conocido. Un chunk recuperado cuenta como relevante si su `metadata.doc_id` coincide con
`documento_id_esperado`.

- **Recall@5**: ¿aparece el documento correcto entre los 5 recuperados? (1 o 0)
- **Precision@5**: qué fracción de los 5 chunks recuperados son del documento correcto
- **MRR** (extra): 1 / posición del primer chunk relevante
- **Hit Rate**: fracción de preguntas con al menos un chunk relevante en el top-5. Con un solo
  documento relevante por pregunta coincide con Recall@5.

| # | Pregunta | Doc. esperado | Chunks del doc. | Hit | Recall@5 | Precision@5 (híbrido) | RR |
|---|---|---|---|---|---|---|---|
| 1 | ¿Cómo permito que un frontend que corre en otro origen llame a mi API desde el navegador? | `cors` | 4 | SÍ | 1 | 0.60 | 1.00 |
| 2 | ¿Cómo envío una notificación por email después de devolver la respuesta, sin que el cliente tenga que esperar? | `background-tasks` | 3 | SÍ | 1 | 0.40 | 1.00 |
| 3 | ¿Qué hace `include_router` y cómo divido mi aplicación en varios archivos? | `bigger-applications` | 14 | SÍ | 1 | 1.00 | 1.00 |
| 4 | ¿Cómo devuelvo un error 404 con `HTTPException` cuando no se encuentra un item? | `handling-errors` | 4 | SÍ | 1 | 0.80 | 1.00 |
| 5 | ¿Cómo creo una dependencia de `Session` de SQLModel para guardar un `Hero` en la base de datos? | `sql-databases` | 8 | SÍ | 1 | 1.00 | 1.00 |

### 6.2 Salida del script

Salida de `python evaluate.py` (completa, con los logs, en
[`evidencia/04_evaluacion.txt`](evidencia/04_evaluacion.txt)).

Detalle por pregunta del modo híbrido (✔ = chunk del documento esperado):

```
[1] ¿Cómo permito que un frontend que corre en otro origen llame a mi API desde el navegador?
    esperado: cors
    top-5:   ['cors', 'handling-errors', 'cors', 'first-steps', 'cors']
    relevantes: ✔ · ✔ · ✔  ->  Hit: SÍ  Recall@5=1  Precision@5=0.60  RR=1.00

[2] ¿Cómo envío una notificación por email después de devolver la respuesta, sin que el cliente tenga que esperar?
    esperado: background-tasks
    top-5:   ['background-tasks', 'background-tasks', 'handling-errors', 'sql-databases', 'middleware']
    relevantes: ✔ ✔ · · ·  ->  Hit: SÍ  Recall@5=1  Precision@5=0.40  RR=1.00

[3] ¿Qué hace include_router y cómo divido mi aplicación en varios archivos?
    esperado: bigger-applications
    top-5:   ['bigger-applications', 'bigger-applications', 'bigger-applications', 'bigger-applications', 'bigger-applications']
    relevantes: ✔ ✔ ✔ ✔ ✔  ->  Hit: SÍ  Recall@5=1  Precision@5=1.00  RR=1.00

[4] ¿Cómo devuelvo un error 404 con HTTPException cuando no se encuentra un item?
    esperado: handling-errors
    top-5:   ['handling-errors', 'handling-errors', 'handling-errors', 'handling-errors', 'body']
    relevantes: ✔ ✔ ✔ ✔ ·  ->  Hit: SÍ  Recall@5=1  Precision@5=0.80  RR=1.00

[5] ¿Cómo creo una dependencia de Session de SQLModel para guardar un Hero en la base de datos?
    esperado: sql-databases
    top-5:   ['sql-databases', 'sql-databases', 'sql-databases', 'sql-databases', 'sql-databases']
    relevantes: ✔ ✔ ✔ ✔ ✔  ->  Hit: SÍ  Recall@5=1  Precision@5=1.00  RR=1.00
```

Métricas por modo y globales:

```
Modo                       Precision@5    Recall@5     MRR
----------------------------------------------------------
BM25 (léxico)                     0.68        1.00    1.00
Vectorial (Pinecone)              0.80        1.00    1.00
Híbrido (Ensemble)                0.76        1.00    1.00
----------------------------------------------------------
Precision@5 máxima alcanzable con este corpus: 0.84 (según cuántos chunks tiene cada documento esperado).

MÉTRICAS GLOBALES DEL HÍBRIDO SOBRE 5 PREGUNTAS (namespace 'dev'):
  • Recall@5 promedio:    1.00 (100%)
  • Precision@5 promedio: 0.76 (76%)
  • Hit Rate:             1.00 (5/5)
  • MRR:                  1.00

Resumen: el recuperador híbrido encontró el documento correcto en 5/5 preguntas (Recall@5=1.00); en promedio 3.8 de cada 5 chunks recuperados son del documento correcto (Precision@5=0.76).
```

### 6.3 Análisis

- **Recall@5 = 1.00 en los tres modos**: en las 5 preguntas, el documento correcto aparece
  en el top-5, y además siempre en la posición 1 (MRR = 1.00).
- **El techo de Precision@5 es 0.84, no 1.00**, porque algunos documentos tienen menos de 5
  chunks: `background-tasks` tiene 3, así que su máximo es 3/5 = 0.60, y `cors` y
  `handling-errors` tienen 4 (máximo 0.80, que `handling-errors` alcanza). `evaluate.py`
  calcula ese techo para no confundir un límite del corpus con un error del recuperador.
- **En español, el vectorial le gana a BM25 (0.80 contra 0.68), y el híbrido queda en el
  medio (0.76).** BM25 compara palabras exactas y el español tiene mucha flexión: "permito"
  no coincide con "permitir" ni "devuelvo" con "devolver". Los embeddings, en cambio,
  capturan esas variantes por significado. Con pesos iguales, los chunks que BM25 trae de
  otros documentos (ej. `handling-errors` en la pregunta de CORS) le restan algo de
  precisión al híbrido.
- **Por qué igual conviene el híbrido.** Ante nombres técnicos exactos, BM25 es más fiable que
  el embedding. En [`evidencia/03_consulta_hibrida.txt`](evidencia/03_consulta_hibrida.txt)
  ("¿Cómo agrego CORSMiddleware con allow_origins?"), 4 de los 5 resultados los trajeron BM25
  y Pinecone a la vez. El Ensemble promueve lo que coincide en ambos rankings y amortigua los
  fallos de cada uno por separado.
- **Mejoras posibles** (con un golden set más grande, para no sobreajustar): *stemming* en
  español en el tokenizer de BM25 (ej. Snowball) y calibrar los pesos del Ensemble sobre un
  conjunto de validación separado.

## 7. Tests y verificaciones

```bash
pytest -v
```

Los 47 tests pasan ([`evidencia/05_tests_pytest.txt`](evidencia/05_tests_pytest.txt)). No usan
red ni API keys: reemplazan Pinecone por objetos fake, así que se pueden correr sin `.env`.

| Archivo | Tests | Qué verifica |
|---|---|---|
| [`tests/test_setup_index.py`](tests/test_setup_index.py) | 4 | Crea el índice si no existe, no lo recrea si existe, detecta mismatch de dimensión y de métrica |
| [`tests/test_ingestion.py`](tests/test_ingestion.py) | 9 | Limpieza de Markdown, chunks de 600 tokens como máximo, esquema de metadata, upsert por lotes con el texto en la metadata, reintentos y rechazo de dimensión incorrecta |
| [`tests/test_rag_system.py`](tests/test_rag_system.py) | 27 | Tokenizer de BM25 (stopwords, tildes, identificadores), BM25 matchea nombres técnicos, corpus reconstruido desde la metadata de Pinecone, `cumple_filtro()` con la sintaxis de Pinecone (13 casos) y `retrieve(filtro=...)`: el filtro llega a Pinecone, BM25 filtra antes de cortar el top-k, sin coincidencias devuelve lista vacía. **Umbral de relevancia:** BM25 descarta los chunks con score 0 y una pregunta fuera del dominio (similitud máxima debajo de `MIN_SIMILITUD`) devuelve lista vacía. **Score combinado:** `retrieve_con_scores()` da el mismo orden que `retrieve()` y un score igual a la fórmula RRF, y respeta el filtro. **Salida JSON:** `resultado_json()` incluye namespace, fuente, `page`, categoría, score, similitud y extracto, y explica por qué no hay resultados cuando la lista viene vacía |
| [`tests/test_evaluate.py`](tests/test_evaluate.py) | 7 | Cálculo de Precision@k, Recall@k y MRR, y validez del golden set (5 preguntas con documentos que existen) |

Verificaciones contra Pinecone real, en [`evidencia/`](evidencia/README.md):

| Evidencia | Qué verifica |
|---|---|
| [`01_ingesta.txt`](evidencia/01_ingesta.txt) | Validación del índice (1536 dims, coseno), 65 chunks subidos |
| [`02_ingesta_idempotente.txt`](evidencia/02_ingesta_idempotente.txt) | La segunda corrida no re-indexa |
| [`03_consulta_hibrida.txt`](evidencia/03_consulta_hibrida.txt) | Top-5 híbrido con namespace, score combinado, fuente, `page`, categoría y el origen de cada resultado |
| [`08_consulta_json.txt`](evidencia/08_consulta_json.txt) | La misma consulta con `--json` |
| [`09_pregunta_fuera_de_dominio.txt`](evidencia/09_pregunta_fuera_de_dominio.txt) | Preguntas fuera de tema (vacaciones, capital de Francia, Nginx): "sin resultados relevantes" en vez de un top-5 |
| [`07_consulta_con_filtros.txt`](evidencia/07_consulta_con_filtros.txt) | La misma consulta sin filtro, con `--categoria` y con un filtro `$in` + `$lte` |
| [`04_evaluacion.txt`](evidencia/04_evaluacion.txt) | Métricas completas |
| [`06_esquema_vector.txt`](evidencia/06_esquema_vector.txt) | Vector real guardado (metadata con texto) y conteo por namespace |

## 8. Checklist de la consigna

Cada requisito de [`preentrega4.md`](preentrega4.md), con el archivo que lo implementa y su
evidencia.

### Componentes obligatorios

| Requisito | Archivo / función | Evidencia |
|---|---|---|
| Pipeline de ingesta en Pinecone Serverless con metadatos (fuente, página, categoría) | [`src/ingestion.py`](src/ingestion.py) → `ingest()` | [`01_ingesta.txt`](evidencia/01_ingesta.txt), [`06_esquema_vector.txt`](evidencia/06_esquema_vector.txt) |
| Recuperador híbrido (vectorial + BM25) | [`src/rag_system.py`](src/rag_system.py) → `RAGSystem` | [`03_consulta_hibrida.txt`](evidencia/03_consulta_hibrida.txt) |
| Script de evaluación con Precision@k y Recall@k sobre un golden set | [`evaluate.py`](evaluate.py) + [`data/golden_set.json`](data/golden_set.json) | [`04_evaluacion.txt`](evidencia/04_evaluacion.txt) |

### Pasos sugeridos

| Paso | Archivo / función | Evidencia |
|---|---|---|
| Índice Serverless con dimensión 1536 | [`src/setup_index.py`](src/setup_index.py) → `ensure_index()` | Log `dimension=1536, metric=cosine` |
| Texto original dentro de la metadata | `ingestion.upsert_chunks()` → `metadata.text` | `06_esquema_vector.txt` |
| `PineconeVectorStore` o SDK nativo | Los dos: SDK nativo para setup, upsert y corpus, y `PineconeVectorStore` para la búsqueda | `rag_system.py` |
| `BM25Retriever` + Pinecone en un `EnsembleRetriever` | `RAGSystem.__init__` | `03_consulta_hibrida.txt` |
| JSON `{"pregunta", "documento_id_esperado"}` y medir el top-5 | [`data/golden_set.json`](data/golden_set.json) · `evaluate.py` | `04_evaluacion.txt` |

### Entregable

| Requisito | Archivo / función | Evidencia |
|---|---|---|
| `.env` con `PINECONE_API_KEY`, `OPENAI_API_KEY`, `INDEX_NAME` | [`.env.example`](.env.example) (el `.env` real está en `.gitignore`) · `config.require()` | Sección [5.3](#53-configuración-de-env) |
| Script de inicialización que crea el índice si no existe | [`src/setup_index.py`](src/setup_index.py) | `01_ingesta.txt` · [`test_setup_index.py`](tests/test_setup_index.py) |
| Dataset de documentación técnica de una librería de Python | [`data/docs/`](data/docs/) (FastAPI, en español) | `12 documentos -> 65 chunks` |
| Chunks con `RecursiveCharacterTextSplitter` | `ingestion.split_documents()` | Log `min=33, prom=427, max=592` tokens |
| Embeddings insertados con contenido y fuente en la metadata | `ingestion.upsert_chunks()` | `06_esquema_vector.txt` |
| Clase `RAGSystem` con `EnsembleRetriever` que devuelve el top-5 | `RAGSystem.retrieve(query, filtro=None)` | `03_consulta_hibrida.txt`, `07_consulta_con_filtros.txt` |
| `evaluate.py` con benchmark de 5 preguntas | [`evaluate.py`](evaluate.py) | `04_evaluacion.txt` |
| Recall@5 | `QueryResult.recall()` | **1.00** |
| Precision@5 | `QueryResult.precision()` | **0.76** (techo 0.84) |
| Reporte en consola | Final de `evaluate.py` | `04_evaluacion.txt` |
| README con pasos para replicar el índice | [Sección 5](#5-cómo-replicar-el-índice-paso-a-paso) | — |

### Errores comunes a evitar

| Error | Cómo se evita | Evidencia |
|---|---|---|
| Mismatch de dimensiones | Dimensión definida una sola vez + `validate_index()` + chequeo en `upsert_chunks()` | Tests `test_detecta_mismatch_de_dimensiones`, `test_upsert_rechaza_dimension_incorrecta` |
| Ignorar el namespace | Todas las operaciones usan `config.NAMESPACE` ([sección 3.4](#34-estrategia-de-segmentación-namespaces-y-metadata)) | `namespaces={'dev': 65}` |
| Subestimar el chunking | 600 tokens con 80 de solapamiento, cortes en encabezados Markdown | Log `min=33, prom=427, max=592` |
