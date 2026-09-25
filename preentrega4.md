Pre-entrega 4: Sistema RAG escalable en la nube con Pinecone
Qué construir
Debes entregar un Módulo de Recuperación Escalable integrado en un repositorio de código. El artefacto principal es un servicio (o conjunto de scripts organizados) en Python que ejecute el flujo completo de un sistema RAG en la nube.

Componentes obligatorios:
Pipeline de Ingesta en Pinecone: Un script que tome un conjunto de documentos (PDFs, Markdown o JSON), los procese y los suba a un índice de Pinecone Serverless utilizando metadatos avanzados (fuente, página, etiquetas de categoría).
Recuperador Híbrido (Hybrid Retriever): Una implementación que combine búsqueda por similitud de vectores con búsqueda léxica (BM25) para mejorar la precisión en términos técnicos o nombres propios.
Script de Evaluación: Una utilidad que calcule al menos dos métricas fundamentales (Precision@k y Recall@k) utilizando un pequeño "Golden Set" de preguntas y respuestas de prueba.
Pasos sugeridos
Preparación de Infraestructura: Crea un índice Serverless en Pinecone (usa la dimensión 1536 si usas OpenAI text-embedding-3-small).
Ingesta Inteligente: No subas el texto plano. Crea un esquema donde guardes el texto original dentro de los metadatos de Pinecone para evitar consultas adicionales a una base de datos relacional.
Configuración de LangChain: Utiliza el PineconeVectorStore de LangChain o el SDK nativo de Pinecone para configurar el motor de búsqueda.
Implementación BM25: Configura un recuperador de LangChain que use BM25Retriever y combínalo con el de Pinecone usando un EnsembleRetriever.
Evaluación Local: Crea un pequeño archivo JSON con pares {"pregunta": "...", "documento_id_esperado": "..."} y mide cuántos de esos documentos aparecen efectivamente en el Top-5 recuperado.
Errores comunes a evitar
Mismatch de Dimensiones: Intentar subir embeddings de 1536 dimensiones a un índice configurado con 512 o 768.
Ignorar el Namespace: En aplicaciones multi-inquilino o con distintos tipos de datos, no usar namespaces en Pinecone hará que la búsqueda sea ruidosa y lenta.
Subestimar el Chunking: Chunks muy pequeños pierden el contexto semántico; muy grandes diluyen la precisión del embedding. Busca un punto medio (~500-800 tokens).
?? Qué entregás y en qué formato

Tipo: ?? Código — un repositorio de GitHub.
Artefacto concreto: repo con el pipeline de ingesta a Pinecone, el recuperador híbrido (BM25 + vectorial) y evaluate.py (Precision@k y Recall@k). El README.md debe incluir los pasos para replicar el índice.
Qué NO hace falta: no hay PDF; el reporte de métricas se imprime en consola y se resume en el README.md.
Repositorio de GitHub que contenga el pipeline de ingesta, el recuperador híbrido y el script de evaluación de métricas. El README debe incluir instrucciones para replicar el índice de Pinecone.
Entregable

Configuración de variables: Crea un archivo .env con PINECONE_API_KEY, OPENAI_API_KEY (o Anthropic) e INDEX_NAME.
Setup de Pinecone: Escribe un script de inicialización que verifique si el índice existe y lo cree si es necesario (modo Serverless).
Pipeline de Ingesta:
Carga un dataset de documentos técnicos (puedes usar la documentación de una librería de Python).
Divide en chunks usando un RecursiveCharacterTextSplitter.
Genera embeddings e insértalos en Pinecone incluyendo el contenido y la fuente en la metadata.
Implementación del Recuperador:
Crea una clase RAGSystem que encapsule un EnsembleRetriever.
El sistema debe recibir una consulta y devolver los top-5 documentos combinando resultados léxicos y semánticos.
Evaluación:
Crea un script evaluate.py.
Define un benchmark de 5 preguntas donde conozcas de antemano el documento fuente.
Ejecuta las consultas y calcula:
Recall@5: ¿Está el documento correcto entre los 5 recuperados?
Precision@5: ¿Qué porcentaje de los 5 recuperados son realmente útiles?
Reporte: Imprime en consola un breve resumen de los resultados de evaluación.