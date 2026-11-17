# Trabajo Práctico N°2 — Sistema de Detección y Clasificación de Razas de Perros

**Tecnicatura Universitaria en Inteligencia Artificial — FCEIA (UNR)**
**Materia:** IA 5.2 Computer Vision — 2° Cuatrimestre 2026

| | |
|---|---|
| **Modalidad** | Grupal (hasta 2 integrantes) |
| **Fecha de entrega** | A definir por la cátedra (vía campus). Solo se consideran los commits hasta esa fecha y hora. |
| **Forma de entrega** | Pull Request **sin mergear** hacia `main` en el repositorio propio del grupo, con el link al PR subido al campus |
| **Repositorio base** | https://github.com/EzequielMatiasArevalo/tuia-dog-recognition-app |
| **Dataset principal** | [70 Dog Breeds Image Dataset (Kaggle)](https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set) |
| **Defensa** | Oral, sobre el trabajo entregado |

---

## Índice

0. [Introducción y objetivo general](#0-introducción-y-objetivo-general)
1. [Paso 0 — Preparar el repositorio (fork + feature branch)](#1-paso-0--preparar-el-repositorio)
2. [Alcance del sistema y lo que provee la cátedra](#2-alcance-del-sistema-y-lo-que-provee-la-cátedra)
3. [Parte 1 — Dataset y preprocesamiento (`1-pre-processing.ipynb`)](#3-parte-1--dataset-y-preprocesamiento)
4. [Etapa 1 — Buscador de imágenes por similitud (`similarity_service.py`)](#4-etapa-1--buscador-de-imágenes-por-similitud)
5. [Etapa 2 — Entrenamiento y comparación de modelos (`2-train.ipynb`)](#5-etapa-2--entrenamiento-y-comparación-de-modelos)
6. [Etapa 3 — Detección y clasificación (`detection_service.py`)](#6-etapa-3--detección-y-clasificación)
7. [Configuración y validación end-to-end](#7-configuración-y-validación-end-to-end)
8. [Entrega: Pull Request](#8-entrega-pull-request)
9. [Buenas prácticas y documentación](#9-buenas-prácticas-y-documentación)
10. [Criterios de evaluación](#10-criterios-de-evaluación)
11. [Recomendaciones generales y errores frecuentes](#11-recomendaciones-generales-y-errores-frecuentes)

---

## 0. Introducción y objetivo general

El objetivo de este trabajo práctico es consolidar y aplicar los conocimientos adquiridos durante las primeras unidades de la materia, integrando conceptos teóricos con la implementación de un sistema completo de **detección y clasificación de razas de perros** con técnicas modernas de Deep Learning.

Se espera que comprendan no solo cómo construir un sistema funcional, sino también **por qué** se toma cada decisión de diseño, analizando los *trade-offs* entre precisión, velocidad de inferencia, consumo de memoria y complejidad computacional.

El sistema evoluciona de forma incremental, desde un buscador por similitud basado en embeddings hasta un pipeline capaz de detectar y clasificar perros en imágenes complejas:

**Embeddings → Búsqueda por similitud → Clasificación → Detección → Pipeline completo**

Al finalizar, el sistema debe ser capaz de:

- Extraer embeddings con modelos preentrenados y construir una base vectorial para recuperar imágenes.
- Identificar la raza de un perro mediante búsqueda por similitud.
- Clasificar razas con modelos entrenados por el grupo.
- Detectar todos los perros de una imagen compleja y clasificar automáticamente cada uno.

El trabajo se organiza en una parte de datos y tres etapas, y cada una construye sobre la anterior:

```text
┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│ 1-pre-processing.ipynb   │──▶│ 2-train.ipynb            │──▶│ src/lib/services/        │
│ EDA, filtrado, conjunto  │   │ NDCG@10 (Etapa 1),       │   │ Etapa 1: similitud       │
│ independiente,           │   │ ResNet fine-tuned vs.    │   │ Etapa 2: clasificación   │
│ augmentation y splits    │   │ CNN propia, métricas     │   │ Etapa 3: detección YOLO  │
│                          │   │ y elección del modelo    │   │ → app Gradio + Docker    │
└──────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘
```

Más allá de "que funcione", **se evalúa el criterio**. Esperamos que cada decisión (filtrado, preprocesamiento, arquitectura, hiperparámetros, umbral, métrica) esté **justificada con evidencia**: gráficos, tablas, números o referencias.

---

## 1. Paso 0 — Preparar el repositorio

### 1.1 Fork del repositorio

1. Un integrante del grupo hace **fork** del repositorio base: https://github.com/EzequielMatiasArevalo/tuia-dog-recognition-app
2. Agrega al otro integrante como colaborador del fork (*Settings → Collaborators*).
3. Clonen **su fork** (no el repositorio de la cátedra):

```bash
git clone git@github.com:<usuario-o-org>/tuia-dog-recognition-app.git
cd tuia-dog-recognition-app
```

> **¿Repositorio privado?** GitHub no permite cambiar a privado un fork de un repositorio público. Si el grupo prefiere un repositorio privado, en lugar de hacer fork **dupliquen** el repositorio:
>
> ```bash
> git clone --bare https://github.com/EzequielMatiasArevalo/tuia-dog-recognition-app.git
> cd tuia-dog-recognition-app.git
> git push --mirror git@github.com:<usuario-o-org>/<repo-privado>.git
> cd .. && rm -rf tuia-dog-recognition-app.git
> ```
>
> Si el repositorio es **privado**, es **obligatorio** agregar a los docentes de la cátedra con permisos de **Maintain** (si el repo está en una organización) o como **colaboradores** (si está en una cuenta personal), para que puedan ver y revisar el PR.
> Un repositorio privado al que los docentes no tengan acceso se considera **NO ENTREGADO**.

### 1.2 Feature branch

Todo el trabajo se hace en una rama propia, **nunca directamente sobre `main`**:

```bash
git checkout -b feature/tp2-razas-perros
git push -u origin feature/tp2-razas-perros
```

Recomendaciones:

- Hagan commits chicos y con mensajes descriptivos (`feat: extract_embedding con ResNet50`, `docs: EDA de imágenes por raza`). El historial también se evalúa: se espera que **ambos integrantes** commiteen y que el trabajo sea incremental, no un único commit final.
- Pueden abrir el Pull Request en modo **Draft** desde el principio e ir pusheando. Al final lo marcan como *Ready for review*.
- `main` debe quedar igual al repositorio base: es la referencia contra la que se revisa el diff.

### 1.3 Levantar el entorno

Sigan las instrucciones del [README.md](README.md) (opción Docker u opción local). Antes de escribir código, verifiquen que:

- El frontend abre en `http://localhost:8080`, con una pestaña por etapa.
- La documentación interactiva del backend abre en `http://localhost:8000/docs` (Swagger, útil para probar endpoints sin el frontend).
- `http://localhost:8000/health` responde `ok` e indica el modelo de embeddings, el modelo YOLO y qué checkpoints encuentra en `models/`.
- PostgreSQL + pgvector levanta (`docker compose up postgres -d`).

> En este punto es esperable que `/health` informe los checkpoints como no encontrados (se generan en la Etapa 2) y que las pestañas del frontend terminen en `failed` con `NotImplementedError`: las funciones de cada etapa todavía no están implementadas.

---

## 2. Alcance del sistema y lo que provee la cátedra

### 2.1 Las tres etapas

| Etapa | Qué hace el sistema | Endpoint | Pestaña del frontend |
|---|---|---|---|
| **1 — Buscador por similitud** | Extrae embeddings, construye la base vectorial, recupera las Top-K imágenes más similares y predice la raza a partir de los vecinos | `POST /search` | Búsqueda por similitud |
| **2 — Clasificación supervisada** | Entrena y compara clasificadores; clasifica la imagen completa con el modelo elegido | `POST /classify` | Clasificación supervisada |
| **3 — Detección y clasificación** | Detecta todos los perros con YOLO, recorta cada detección y la clasifica | `POST /detect` | Detección y clasificación |

### 2.2 Cómo funciona la app

```text
Frontend (Gradio :8080)
   │  POST /upload   → guarda la imagen y devuelve su path
   │  POST /search   → {source_path, model, top_k} ─┐
   │  POST /classify → {source_path, model}         ├─▶ HTTP 202 + job_id (asincrónico)
   │  POST /detect   → {source_path}               ─┘
   │  GET  /status/{job_id} → done | inProgress | failed + links al resultado
   ▼
Backend (FastAPI :8000)
   ├─ SimilarityService  (Etapa 1) ──▶ EmbeddingStore (PostgreSQL + pgvector, o JSON)
   ├─ ClassifierService  (Etapa 2) ──▶ models/*.pth | models/*.onnx
   └─ DetectionService   (Etapa 3) ──▶ YOLO + ClassifierService
```

### 2.3 Lo que ya está resuelto

La cátedra provee un repositorio base completamente funcional. **No hace falta desarrollar** interfaces gráficas, APIs, contenedores Docker ni configuración de infraestructura:

- Docker y Docker Compose configurados (backend, frontend y PostgreSQL + pgvector).
- Aplicación Gradio con una pestaña por etapa.
- API asincrónica (`/upload`, `/search`, `/classify`, `/detect`, `/status`, `/models`, `/health`) y manejo de jobs.
- Base vectorial configurable: PostgreSQL + pgvector o archivo JSON (`USE_PGVECTOR`).
- Orquestación de cada etapa: `search` e `index_image` (Etapa 1), `classify_image` (Etapa 2) y `predict` (Etapa 3).
- Carga de checkpoints `.pth` / `.onnx` y selección del modelo activo (`ClassifierService.load_model`, `set_active_model`).
- Scripts: descarga del dataset, indexación de la base vectorial y entrenamiento/evaluación del clasificador (`scripts/`).
- Funciones auxiliares de evaluación (`src/lib/evaluation/metrics.py`: NDCG@k, precision/recall/F1, specificity) y de visualización (`src/lib/visualization/draw.py`).

### 2.4 Funciones a implementar

Hoy todas lanzan `NotImplementedError`. Cada una tiene un docstring con el contrato esperado:

| Etapa | Función | Archivo |
|---|---|---|
| 1 | `extract_embedding(image)` | `src/lib/services/similarity_service.py` |
| 1 | `search_similar_images(embedding, top_k)` | `src/lib/services/similarity_service.py` |
| 1 | `predict_breed_from_neighbors(results)` | `src/lib/services/similarity_service.py` |
| 2 | `train_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `evaluate_classifier()` | `src/lib/services/classifier_service.py` |
| 2 | `extract_custom_embedding(image)` | `src/lib/services/classifier_service.py` |
| 3 | `detect_dogs(image)` | `src/lib/services/detection_service.py` |
| 3 | `classify_detected_dog(crop)` | `src/lib/services/detection_service.py` |

### 2.5 Reglas

- En las **Etapas 1 y 3** el trabajo se limita a completar las funciones indicadas (pequeñas y puntuales), que se integran con la infraestructura provista.
- En la **Etapa 2**, además de las funciones, el grupo **entrena sus propios modelos** y realiza un **estudio de resultados** (métricas, comparaciones y análisis de errores) en `2-train.ipynb`.
- Pueden agregar módulos auxiliares (por ejemplo, `src/lib/models/cnn_custom.py` con la arquitectura de la CNN propia, o un módulo de preprocesamiento compartido) y métodos privados dentro de los servicios. **Cualquier otra modificación fuera de las funciones indicadas** debe justificarse en la descripción del PR (ver [8.3](#83-descripción-del-pr)).
- La evaluación se hace ejecutando el proyecto **exclusivamente con los comandos provistos** en el repositorio (`docker compose build` y `docker compose up`).
- Se recomienda mantener cada componente desacoplado para facilitar el testing, el debugging y la extensibilidad.

---

## 3. Parte 1 — Dataset y preprocesamiento

**Archivo:** `1-pre-processing.ipynb` (o en Google Colab, ver [8.2](#82-notebooks-repositorio-o-google-colab))

Todo el tratamiento del dataset se hace y se documenta en esta notebook. Tiene que poder leerse como un informe: celdas Markdown que explican qué se hace y por qué, código, y salidas (gráficos y tablas) **ejecutadas y guardadas**. Una notebook sin salidas visibles se considera **NO ENTREGADA**.

### 3.1 Dataset

Se utiliza el **70 Dog Breeds Image Dataset** (Kaggle), que se descarga con `python scripts/download_dataset.py` y viene dividido en `train/`, `valid/` y `test/`, con una carpeta por raza.

**Se espera:**

- Describir el dataset: cantidad de razas, cantidad de imágenes por raza en cada split, formato, resolución y licencia de uso.
- Analizar cómo están encuadradas las imágenes (¿el perro ocupa casi toda la imagen? ¿hay fondos, personas u otros perros?) y compararlo con lo que va a recibir el clasificador en la Etapa 3: **recortes producidos por YOLO**, con otro encuadre, resolución y fondo. Esa diferencia es un cambio de dominio que hay que medir (ver [3.4](#34-conjunto-independiente-de-evaluación)).
- Describir explícitamente las **limitaciones** del dataset (razas muy parecidas entre sí, sesgos de las imágenes de origen web, ausencia de perros mestizos, etc.) y cómo pueden afectar los resultados.

### 3.2 Análisis exploratorio (EDA)

Como mínimo:

- **Distribución de clases**: cantidad de imágenes por raza en cada split, con gráfico y estadísticas (media, mediana, mínimo, máximo). Discutir el **desbalance** y si el tamaño de `valid` y `test` alcanza para obtener métricas por raza estables.
- **Resoluciones y formato**: distribución de ancho × alto, relación de aspecto, canales (RGB / escala de grises), extensiones y archivos corruptos o que no abren.
- **Muestra visual**: grilla con ejemplos por raza, incluyendo pares de razas fáciles de confundir.
- **Variaciones**: iluminación (histograma de brillo / contraste), pose, fondo, oclusiones y cantidad de perros por imagen. Al menos una parte puede ser cualitativa.

### 3.3 Control de calidad y filtrado

Definan **reglas de filtrado explícitas y reproducibles** y reporten cuántas imágenes descarta cada una:

- Archivos corruptos o que no se pueden leer.
- Imágenes con resolución menor a un mínimo.
- Imágenes **borrosas** (por ejemplo, varianza del Laplaciano bajo un umbral) o con exposición extrema.
- **Duplicados y casi-duplicados** (hash perceptual o similitud de embeddings > 0.95), tanto dentro de un split como **entre splits**: un duplicado entre `train` y `test` es fuga de datos.
- Imágenes mal etiquetadas o donde no hay un perro visible (opcional: pueden usar el detector YOLO de la Etapa 3 para encontrarlas).

Muestren ejemplos de imágenes descartadas por cada regla: así se valida que el umbral elegido tiene sentido. Documenten si el filtrado se aplica a todos los splits o solo a `train`, y por qué.

### 3.4 Conjunto independiente de evaluación

Construyan un conjunto de imágenes **que no pertenezcan al dataset** (descargadas de internet o fotos propias), etiquetadas con las mismas razas que las carpetas del dataset. Es el que mide cómo se comporta el sistema "en el mundo real" y se usa en las tres etapas:

- Imágenes "limpias" (un perro, bien encuadrado) para evaluar las Etapas 1 y 2.
- Escenas con **varios perros**, personas u otros objetos para la Etapa 3, e imágenes **sin perros** (por ejemplo, gatos) para verificar falsos positivos del detector.
- Como referencia, apunten a **al menos 100 imágenes** que cubran **20 razas o más**, incluyendo razas fáciles de confundir.
- Documenten la fuente y la licencia de cada imagen. Guárdenlas fuera del control de versiones (por ejemplo, en `data/dataset/independent/<raza>/`, que ya está en el `.gitignore`) y compártanlas con un link de solo lectura para los docentes.

### 3.5 Preprocesamiento

Definan y justifiquen:

- **Resize** a `IMAGE_SIZE` (224 por defecto): resize directo, resize + *center crop* o *padding* para preservar la relación de aspecto.
- **Normalización**: estadísticas de ImageNet para los modelos preentrenados; para la CNN propia, justifiquen si usan las de ImageNet o las calculadas sobre `train`.
- **Orden de canales**: la app entrega las imágenes en **BGR** (OpenCV), mientras que PIL y torchvision trabajan en RGB.

> ⚠️ **Muy importante:** el **mismo pipeline** de preprocesamiento (tamaño, orden de canales BGR/RGB, escala de píxeles, media/desvío) debe usarse en las notebooks **y** en los servicios de la app. Es la causa número uno de que "el modelo anda en la notebook pero no en la app". Encapsúlenlo en funciones reutilizables dentro de un módulo de `src/lib/` e impórtenlo desde ambos lados.

### 3.6 Data augmentation

`albumentations` ya está en `requirements.txt` (también pueden usar `torchvision.transforms.v2`).

- Incluyan, como mínimo: **horizontal flip**, **rotación**, **blur**, **variaciones de brillo**, **variaciones de contraste** y **ruido**. Pueden sumar otras (*random resized crop*, *color jitter*, *coarse dropout*) justificándolas.
- Justifiquen cada transformación y sus rangos, y **eviten las que no ocurren en la realidad o que cambian la raza aparente** (flips verticales, rotaciones de 90°, cambios de color extremos que alteran el pelaje).
- Muestren una grilla de ejemplos aumentados de una misma imagen.
- La augmentation se aplica **solo sobre el conjunto de entrenamiento**, nunca sobre validación, test ni el conjunto independiente.
- Si la usan para mitigar el desbalance, explíquenlo y compárenlo con otras alternativas (*weighted sampling*, *class weights*).

### 3.7 Partición de datos (splits)

- Definan **train / validation / test**: pueden usar los splits del dataset o redefinirlos, justificando la decisión y las proporciones.
- En la Etapa 1, `train` es la **galería** que se indexa en la base vectorial, y `test` + el conjunto independiente son las **consultas**. Una imagen de consulta nunca debe estar indexada.
- Verifiquen que no haya **fuga de datos** (la misma imagen o un casi-duplicado en más de un split).
- Fijen semillas y guarden los splits finales (CSV/JSON con rutas y etiquetas) para que `2-train.ipynb` y `train_classifier` los reutilicen.

### 3.8 Licencias y uso de imágenes

Incluyan una sección breve con la licencia del dataset y si permite el uso que le dan, la fuente y licencia de las imágenes del conjunto independiente, y el criterio aplicado a imágenes donde aparecen personas identificables (evitarlas o no publicarlas). No suban imágenes al repositorio: el `.gitignore` ya excluye `data/dataset/*`.

### 3.9 Salidas esperadas de la notebook 1

- [ ] Descripción del dataset y de sus limitaciones.
- [ ] EDA: distribución de clases, resoluciones, ejemplos visuales y variaciones.
- [ ] Reglas de calidad con cantidad de imágenes descartadas por regla y ejemplos.
- [ ] Conjunto independiente construido y documentado (fuente, licencia, razas e imágenes por raza).
- [ ] Preprocesamiento justificado y encapsulado en un módulo reutilizable.
- [ ] Pipeline de augmentation justificado y visualizado.
- [ ] Splits guardados y verificación de ausencia de fuga de datos.
- [ ] Tabla resumen final: cantidad de razas e imágenes por split (incluido el conjunto independiente).
- [ ] Sección de licencias.

---

## 4. Etapa 1 — Buscador de imágenes por similitud

**Archivo principal:** `src/lib/services/similarity_service.py` · **Evaluación:** `2-train.ipynb`

### 4.1 Cómo funciona

- **Indexar** (`scripts/build_index.py`): para cada imagen de la galería, `index_image` ejecuta `extract_embedding` y guarda un `EmbeddingRecord` en la base vectorial con la estructura `id_imagen`, `embedding`, `path`, `breed` y `metadata`. El `path` se guarda relativo a `DATA_PATH` (por ejemplo, `dataset/train/<raza>/<imagen>.jpg`), así la misma base sirve en local y en Docker; para leer la imagen de un registro usen `resolve_path(record.path)`.
- **Buscar** (`search`, pestaña de la Etapa 1): `_load_image` → `extract_embedding` (o `extract_custom_embedding` si se eligió un modelo entrenado) → `search_similar_images` → `predict_breed_from_neighbors`. El resultado se escribe como JSON y el frontend muestra la **imagen consultada**, las **Top-K imágenes similares** (10 por defecto) y la **raza predicha**.

La base vectorial provista es **PostgreSQL + pgvector** (por defecto) o un **archivo JSON** (`USE_PGVECTOR=false`). Si prefieren otra opción (ChromaDB, FAISS, MongoDB), implementen un *store* con la misma interfaz (`all`, `append` y, opcionalmente, `search`) y justifíquenlo en el PR.

### 4.2 Funciones a implementar

#### `extract_embedding(self, image) -> list[float]`

- **Entrada:** imagen `np.ndarray` en **BGR uint8** (como la devuelve `cv2.imread`).
- **Salida:** `list[float]` de longitud **exactamente `EMBEDDING_DIM`** (la tabla de pgvector se crea como `vector(EMBEDDING_DIM)`).
- **Modelo:** preentrenado en ImageNet, **sin la capa de clasificación** (por ejemplo, `model.fc = nn.Identity()` en torchvision o `num_classes=0` en `timm`). Opciones y dimensión del embedding resultante:

| Modelo | Dimensión |
|---|---|
| ResNet18 | 512 |
| ResNet50 | 2048 |
| EfficientNet-B0 | 1280 |
| ConvNeXt-Tiny | 768 |

- **Recomendaciones:**
  - BGR→RGB, resize a `self.image_size`, normalización de ImageNet, `HWC→CHW` y dimensión de batch.
  - `model.eval()` y `torch.no_grad()`.
  - **Normalicen L2** el embedding y devuelvan floats de Python (`emb.astype(float).tolist()`), no tensores ni `np.float32`.
  - Carguen el modelo **una sola vez** (por ejemplo, cacheándolo en `self` en la primera llamada), no en cada imagen.
  - Los pesos preentrenados se descargan en el primer uso: en Docker, eso ocurre dentro del contenedor y requiere conexión a internet.

#### `search_similar_images(self, embedding, top_k) -> list[Neighbor]`

- **Salida:** lista de `Neighbor(path, breed, score)` de largo ≤ `top_k`, **ordenada por score descendente** (mayor score = más similar). No completen `url`: la orquestación lo agrega.
- **Recomendaciones:**
  - Con pgvector, `self.store.search(embedding, top_k)` resuelve la búsqueda en la base con el operador `<=>` (distancia coseno). Con JSON, recorran `self.store.all()`.
  - Calculen el score con `self.similarity(...)`, que respeta `SIMILARITY_METRIC` (`cosine` o `l2`; para `l2` devuelve `1 / (1 + distancia)`). Si usan `l2` con pgvector, tengan en cuenta que `store.search` ordena por distancia coseno.
  - Ambas métricas deben estar soportadas y compararse en la evaluación.

#### `predict_breed_from_neighbors(self, results) -> tuple[str, float]`

- **Salida:** `(raza, score)`.
- **Recomendaciones:**
  - Voto mayoritario entre las razas de los vecinos, opcionalmente **ponderado por score**. Documenten qué devuelven como score (por ejemplo, la similitud media de los vecinos de la raza ganadora o la proporción de votos).
  - Si no hay vecinos o el mejor score queda por debajo de `SIMILARITY_THRESHOLD`, devuelvan `"unknown"`.
  - **Calibren el umbral con datos de validación**, no a ojo, y configúrenlo en el `.env`.

### 4.3 Evaluación (en `2-train.ipynb`)

- **Galería:** las imágenes de `train`. **Consultas:** `test` y el conjunto independiente (nunca indexadas).
- Para cada consulta: ejecutar la búsqueda, recuperar las **Top-10** y calcular **NDCG@10** con relevancia binaria (`rel_i = 1` si el vecino `i` es de la misma raza que la consulta):

  `NDCG@10 = DCG@10 / IDCG@10`, con `DCG@10 = Σ rel_i / log2(i + 1)` para `i = 1..10`.

  `IDCG@10` es el DCG del ranking ideal: si la raza tiene al menos 10 imágenes en la galería, el ideal son 10 vecinos relevantes. No se calcula reordenando los resultados devueltos (con ese atajo, un único acierto en la primera posición daría NDCG = 1). `ndcg_at_k(relevancias, k=10, n_relevant=...)` de `src/lib/evaluation/metrics.py` implementa esta definición, con `n_relevant` = cantidad de imágenes de la raza en la galería.
- Reporten el **NDCG@10 medio** (y su distribución o el valor por raza), la **precision@10** y la **accuracy** de la raza predicha, **por separado** para `test` y para el conjunto independiente.
- Comparen `cosine` vs. `l2` y analicen el efecto de `TOP_K` y del umbral.
- **Justifiquen el resultado**: qué razas fallan y por qué (razas parecidas, fondo, pose, varios perros), con ejemplos visuales de consultas y sus Top-10.

> **Sugerencia:** para evaluar en la notebook no hace falta la base de datos. Calculen los embeddings de la galería en una matriz de `numpy` y busquen con un producto matricial. `EmbeddingStore.append` (JSON) reescribe el archivo completo en cada llamada, por lo que es lento para miles de imágenes.

---

## 5. Etapa 2 — Entrenamiento y comparación de modelos

**Archivos:** `2-train.ipynb` (o en Google Colab, ver [8.2](#82-notebooks-repositorio-o-google-colab)) y `src/lib/services/classifier_service.py`

El entrenamiento se recomienda en **Google Colab** con GPU (*Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU*). La notebook ya tiene las celdas para clonar el fork, instalar dependencias y descargar el dataset. Hay que comparar **dos** modelos, **ambos obligatorios**, sobre **el mismo conjunto de test** y con **las mismas métricas**:

| | Modelo A — Fine-tuning | Modelo B — CNN propia |
|---|---|---|
| **Qué es** | ResNet18 o ResNet50 preentrenada en ImageNet, ajustada a las razas del dataset | Una CNN diseñada y entrenada desde cero por el grupo |
| **Nombre en la app** | `resnet18_finetuned` (`RESNET18_MODEL_NAME`) | `cnn_custom` (`CNN_CUSTOM_MODEL_NAME`) |
| **Objetivo** | Línea de base fuerte con *transfer learning* | Entender qué se logra con un modelo propio y datos limitados |

Si eligen ResNet50 para el Modelo A, mantengan el nombre lógico `resnet18_finetuned` en la app, apunten `RESNET18_MODEL_NAME` a su checkpoint y documéntenlo.

### 5.1 Funciones a implementar

La notebook solo orquesta estas funciones: `classifier.set_active_model(...)` elige el modelo y luego se llama a `train_classifier()` y `evaluate_classifier()`.

#### `train_classifier(self) -> None`

- Entrena el modelo activo (`self.active_model_name`) sobre `self.dataset_path`, con los splits de la Parte 1 y el preprocesamiento y augmentation justificados.
- Guarda el checkpoint en `self.active_checkpoint` (por ejemplo, `models/resnet18_finetuned.pth`), en el formato que espera `load_model` (ver [5.6](#56-exportación-de-los-modelos)).
- **Recomendaciones:**
  - Guarden el **mejor checkpoint según validación** (*early stopping*), no el de la última época.
  - Guarden el historial por época (loss y accuracy de train y validación), por ejemplo en un JSON dentro de `self.output_path`, para graficar las curvas en la notebook.
  - Fijen semillas. Si parametrizan hiperparámetros, háganlo por configuración, no hardcodeados.

#### `evaluate_classifier(self) -> dict[str, float]`

- Evalúa el modelo activo sobre el conjunto de **test**.
- Devuelve un diccionario con `accuracy`, `precision`, `recall` (sensibilidad), `specificity` (especificidad) y `f1`.
- Con 70 clases, precision, recall y F1 se promedian por clase (*macro*). La especificidad de cada clase se calcula *one-vs-rest* (`TN / (TN + FP)`) y luego se promedia. Pueden usar las funciones de `src/lib/evaluation/metrics.py`.

#### `extract_custom_embedding(self, image) -> list[float]`

- **Entrada:** imagen BGR uint8. **Salida:** `list[float]` de longitud `EMBEDDING_DIM`, normalizada L2.
- Embedding de la **capa previa al clasificador** del modelo activo (ResNet fine-tuned o CNN propia).
- Lo usan la búsqueda por similitud cuando se elige un modelo entrenado (campo `model` de `/search`) y `scripts/build_index.py` cuando `EMBEDDING_MODEL` no es `baseline`.
- Debe replicar exactamente el preprocesamiento del entrenamiento.

### 5.2 Modelo A — Fine-tuning de ResNet

- Reemplacen la capa final por una capa lineal con una salida por raza.
- Elijan y justifiquen la estrategia: *feature extraction* (backbone congelado, solo se entrena la cabeza), *fine-tuning* completo o descongelado gradual, con *learning rates* diferenciados por capa. Se valora comparar al menos dos estrategias.
- Documenten los hiperparámetros: optimizador, *learning rate* y *scheduler*, *batch size*, épocas y regularización (*weight decay*, *dropout*, *label smoothing*).

### 5.3 Modelo B — CNN propia

- **Arquitectura:** capas, cantidad de parámetros, dimensión del embedding y diagrama o `summary`. Justifiquen decisiones como profundidad, BatchNorm, Dropout y *global pooling*.
- Incluyan una capa de embedding de dimensión `EMBEDDING_DIM` antes del clasificador, para poder usarla en la búsqueda por similitud.
- **Experimentos:** qué probaron, qué no funcionó y por qué. Una tabla de experimentos es suficiente.
- Es esperable que rinda menos que el Modelo A: el análisis de **por qué** (datos, capacidad, ausencia de preentrenamiento) es parte central de la nota.

> **Sugerencias:** arranquen con algo chico (4–6 bloques convolucionales), usen *early stopping* y guarden el mejor checkpoint según validación. Si no tienen GPU, usen Google Colab o Kaggle Notebooks.

### 5.4 Evaluación

Ambos modelos se evalúan con el mismo protocolo:

**a) Métricas de clasificación** — accuracy, precision, recall (sensibilidad), specificity (especificidad) y F1 sobre `test`, más una tabla por raza con las de peor desempeño.

**b) Matriz de confusión** — con 70 razas, muestren la matriz normalizada y la lista de los pares de razas más confundidos.

**c) Curvas de entrenamiento** — loss y accuracy de train vs. validación, con discusión de *overfitting* / *underfitting*.

**d) Conjunto independiente** — las mismas métricas sobre el conjunto de la Parte 1. Comparen con `test` y discutan el cambio de dominio.

**e) Embeddings en la búsqueda** — NDCG@10 de la Etapa 1 usando los embeddings del baseline, del Modelo A y del Modelo B, con la misma galería y las mismas consultas. Visualicen los embeddings de `test` de cada modelo con **PCA o t-SNE**, lado a lado y coloreados por raza (con un subconjunto de razas para que se pueda leer).

**f) Análisis de errores** — ejemplos de **falsos positivos** y **falsos negativos** con la imagen y su score. Formulen hipótesis sobre las causas (razas parecidas, pose, fondo, varios perros, etiqueta errónea).

**g) Costo computacional** — cantidad de parámetros, tamaño en MB, tiempo de inferencia por imagen en CPU y dimensión del embedding.

### 5.5 Comparación y elección final

Cierren la notebook con:

- Una **tabla comparativa** de todas las métricas anteriores para ambos modelos (y el baseline en la Etapa 1).
- Una **conclusión justificada**: qué modelo usa la app para clasificar (Etapas 2 y 3) y qué embeddings usa para la búsqueda (`EMBEDDING_MODEL`), considerando desempeño, costo y facilidad de integración.
- Una comparación entre **búsqueda por similitud** y **clasificación supervisada**: ventajas, limitaciones y en qué casos conviene cada una.

### 5.6 Exportación de los modelos

El backend carga los checkpoints con `ClassifierService.load_model`, que admite:

- **`.pth`**: se carga con `torch.load(path, map_location="cpu", weights_only=False)`. Esto implica guardar el **modelo completo** (`torch.save(model, path)`), no solo el `state_dict`, y que **la clase del modelo sea importable** desde el backend con el mismo módulo y nombre que tenía al guardarse. Si definen la CNN propia dentro de la notebook, el `torch.load` va a fallar en la app: pónganla en un módulo (por ejemplo, `src/lib/models/cnn_custom.py`) e impórtenla desde la notebook.
- **`.onnx`**: se carga con `onnxruntime.InferenceSession` y no depende del código de la clase. Exporten con `torch.onnx.export` y actualicen la extensión en `RESNET18_MODEL_NAME` / `CNN_CUSTOM_MODEL_NAME`.

La app necesita traducir el índice predicho a un nombre de raza: **guarden la lista de razas junto con el modelo** (por ejemplo, como atributo del modelo antes de `torch.save`, o en la metadata del ONNX) en lugar de depender de que el dataset esté presente en el contenedor.

> Los archivos `*.pth`, `*.pt` y `*.onnx` están en el `.gitignore` y **no deben subirse al repositorio**. Súbanlos a Google Drive, Hugging Face Hub u otro servicio con un **link público de solo lectura**, y pongan el link en el README y en el PR.

### 5.7 Salidas esperadas de la notebook 2

- [ ] Etapa 1: NDCG@10, precision@10 y accuracy del baseline sobre `test` y el conjunto independiente, con `cosine` y `l2`.
- [ ] Modelo A: estrategia de fine-tuning, hiperparámetros y curvas de entrenamiento.
- [ ] Modelo B: arquitectura, hiperparámetros, curvas de entrenamiento y tabla de experimentos.
- [ ] Métricas de ambos modelos (accuracy, precision, recall, specificity, F1) y matriz de confusión.
- [ ] Evaluación sobre el conjunto independiente.
- [ ] NDCG@10 con los embeddings de cada modelo y visualización PCA/t-SNE.
- [ ] Análisis de errores con ejemplos visuales.
- [ ] Costo computacional.
- [ ] Tabla comparativa final y elección justificada.
- [ ] Checkpoints exportados y links de descarga.

---

## 6. Etapa 3 — Detección y clasificación

**Archivo principal:** `src/lib/services/detection_service.py`

### 6.1 Cómo funciona

El flujo esperado, ya orquestado por `DetectionService.predict`, es:

1. El usuario carga una imagen.
2. YOLO detecta todos los perros (`detect_dogs`).
3. Se generan las *bounding boxes* (recortadas a los bordes de la imagen con `_clip_xyxy`).
4. Se recortan las regiones detectadas.
5. Cada recorte se clasifica (`classify_detected_dog`) con el modelo activo de la Etapa 2.
6. Se generan las predicciones finales: el frontend muestra la imagen original con las *bounding boxes*, la raza predicha y los scores de confianza.

`classify_image` (pestaña de la Etapa 2) también usa `classify_detected_dog`, tratando la imagen completa como un recorte: esa pestaña necesita esta función implementada.

### 6.2 Funciones a implementar

#### `detect_dogs(self, image) -> list[tuple[tuple[int, int, int, int], float]]`

- **Entrada:** imagen BGR uint8.
- **Salida:** lista de `((x1, y1, x2, y2), confianza)` en **píxeles enteros** de la imagen original. Si no hay perros, una lista vacía (no `None` ni excepción).
- **Recomendaciones:**
  - Usen un modelo **YOLO preentrenado en COCO** (por ejemplo, YOLOv8n o YOLO11n vía `ultralytics`). **No es necesario entrenar el detector.**
  - `self.yolo_model_name`, `self.conf_threshold` y `self.dog_class_id` (clase `dog` = 16 en COCO) vienen del `.env`: filtren por esa clase y por confianza sin hardcodear valores.
  - Carguen el detector **una sola vez**, no en cada llamada.
  - `ultralytics` descarga los pesos en el primer uso, en la ruta de `YOLO_MODEL` (por defecto `models/yolov8n.pt`, que persiste fuera del contenedor). Si cambian de modelo, mantengan la ruta dentro de `models/`.

#### `classify_detected_dog(self, crop) -> tuple[str, float]`

- **Entrada:** recorte BGR uint8 (puede ser chico o muy alargado).
- **Salida:** `(raza, score)`, donde `score` es la probabilidad (*softmax*) de la raza predicha.
- **Recomendaciones:**
  - Usen `self.classifier.load_model()` (modelo activo, ya cacheado) y **el mismo preprocesamiento del entrenamiento**.
  - Con `.pth`: `model.eval()` y `torch.no_grad()`. Con `.onnx`: `model.run(None, {input_name: tensor})`.
  - Traduzcan el índice a raza con la lista guardada junto al modelo.
  - Opcional: devolver `"unknown"` si la probabilidad queda por debajo de un umbral configurable en el `.env` (justifiquen la variable agregada).

### 6.3 Validación

El sistema debe funcionar con:

- **Un perro.**
- **Múltiples perros**, incluso superpuestos.
- **Escenas complejas** con personas, otros animales u objetos.
- **Imágenes sin perros** (deben devolver una lista vacía) y con gatos u otros animales parecidos (verificar falsos positivos).

Muestren el efecto de 2 o 3 valores de `YOLO_CONF_THRESHOLD` y justifiquen el elegido.

---

## 7. Configuración y validación end-to-end

### 7.1 Configuración

**No se permite hardcodear configuraciones**: todo lo configurable va por variables de entorno (`.env`) y `src/lib/config.py`.

| Variable | Para qué |
|---|---|
| `EMBEDDING_MODEL` | Modelo de embeddings por defecto: `baseline`, `resnet18_finetuned` o `cnn_custom` |
| `EMBEDDING_DIM` | Dimensión del embedding del modelo elegido |
| `IMAGE_SIZE` | Tamaño de entrada de los modelos |
| `SIMILARITY_METRIC` / `SIMILARITY_THRESHOLD` | Métrica (`cosine` o `l2`) y umbral de similitud (Etapa 1) |
| `TOP_K` | Cantidad de vecinos a recuperar |
| `MODEL_PATH`, `RESNET18_MODEL_NAME`, `CNN_CUSTOM_MODEL_NAME` | Ubicación de los checkpoints (Etapa 2) |
| `TORCH_HOME` | Caché de los pesos preentrenados de torchvision (`models/torch`) |
| `YOLO_MODEL`, `YOLO_CONF_THRESHOLD`, `YOLO_DOG_CLASS_ID` | Configuración de YOLO (Etapa 3) |
| `USE_PGVECTOR` y `POSTGRES_*` | Base vectorial |
| `DATASET_PATH`, `EMBEDDINGS_PATH`, `OUTPUT_PATH` | Paths |

- Actualicen `.env.docker.example` y `.env.local.example` con los valores elegidos (por ejemplo, el umbral calibrado).
- Si agregan una variable al `.env`, declárenla también en `Settings` (`src/lib/config.py`) para leerla desde el código; las variables no declaradas se ignoran.
- Si cambian `EMBEDDING_DIM` y ya existía la tabla, el *store* la recrea (y **borra** los registros previos): hay que reindexar.
- La base vectorial guarda los embeddings de **un modelo a la vez**. El selector de modelo de la pestaña de la Etapa 1 (campo `model` de `/search`) cambia el extractor de la consulta, así que la base debe estar indexada con ese mismo modelo y su dimensión: ajusten `EMBEDDING_MODEL` y `EMBEDDING_DIM` y reindexen con `scripts/build_index.py`. Documenten con qué modelo queda indexada la base entregada.
- Si agregan dependencias, agréguenlas a `requirements.txt` con versión fija y reconstruyan la imagen (`docker compose build`).

### 7.2 Validación end-to-end

Demuestren que la app funciona:

1. Indexen la galería (`train`) en la base vectorial: `docker compose exec backend python /scripts/build_index.py --split train` en Docker, o `python scripts/build_index.py --split train` en local (ambos usan el mismo PostgreSQL).
2. **Etapa 1:** busquen con imágenes del conjunto independiente (no indexadas), con el baseline y con un modelo entrenado. Se debe ver la imagen consultada, las Top-10 y la raza predicha.
3. **Etapa 2:** clasifiquen imágenes con ambos modelos entrenados.
4. **Etapa 3:** procesen imágenes con un perro, con varios perros, con una escena compleja y sin perros.
5. Incluyan en el PR **capturas de pantalla** de cada pestaña.
6. Prueben casos borde: imagen sin perros, checkpoint faltante (debe fallar con un mensaje claro en `/status`) e imagen que no se puede leer.

Todo debe funcionar partiendo de un clon limpio, siguiendo **solo** las instrucciones del PR, con `docker compose build` y `docker compose up`.

### 7.3 Mejoras opcionales (suman puntos)

- Usar `PgVectorEmbeddingStore.search` (búsqueda en la base con `<=>`) en lugar de recorrer todos los registros en Python.
- Que el selector de modelo de la Etapa 1 funcione sin reindexar (una base o tabla por modelo).
- Tests unitarios básicos (`pytest`) de las funciones implementadas: dimensión y norma del embedding, orden de los vecinos, casos vacíos.
- Exportar los modelos a ONNX y comparar la latencia en CPU contra `.pth`.
- Comparar YOLOv8n vs. YOLO11n en detecciones y latencia en CPU.
- Visualizar con Grad-CAM qué regiones usa el clasificador.

---

## 8. Entrega: Pull Request

La entrega es un **Pull Request desde `feature/...` hacia `main` de su propio repositorio**, **SIN MERGEAR**. El PR se revisa y comenta sobre el diff. **No abran el PR contra el repositorio de la cátedra.** Suban el link al PR en el campus antes de la fecha de entrega.

### 8.1 Contenido del PR

El diff debe incluir:

- `1-pre-processing.ipynb` y `2-train.ipynb` (ver [8.2](#82-notebooks-repositorio-o-google-colab)). Completen los nombres de los integrantes en la celda *Equipo*.
- `similarity_service.py`, `classifier_service.py` y `detection_service.py` con las funciones implementadas, más cualquier módulo auxiliar que hayan agregado.
- Archivos `.env.*.example` actualizados.
- `requirements.txt` actualizado (si corresponde).
- `README.md` actualizado (ver [9.2](#92-documentación)).

No debe incluir: datasets ni el conjunto independiente, pesos de modelos (`.pth`/`.pt`/`.onnx`), archivos `.env` con secretos, `output/` ni `data/embeddings.json` con embeddings generados.

### 8.2 Notebooks: repositorio o Google Colab

Las notebooks se pueden entregar de dos formas:

1. **Commiteadas en el repositorio**, reemplazando `1-pre-processing.ipynb` y `2-train.ipynb`.
2. **En Google Colab.** En este caso, las notebooks `1-pre-processing.ipynb` y `2-train.ipynb` del repositorio igual deben commitearse y funcionan como **notebooks de referencia**. Cada una debe contener, en una celda Markdown al inicio, el **link a la notebook de Colab correspondiente**. Ese link debe tener **permiso de lectura para los docentes**: compártanlo con sus cuentas o con "Cualquier persona con el enlace" como *Lector*.

En **ambos casos**, las notebooks deben entregarse **ejecutadas y con la salida del código visible**: gráficos, tablas, métricas y logs de entrenamiento.

> ⚠️ **Se considera NO ENTREGADO:**
> - Una notebook sin ejecutar o sin las salidas visibles (por ejemplo, guardada con *"Clear all outputs"* o con la opción de Colab que omite las salidas al guardar).
> - Una notebook en Colab cuyo link no figure en la notebook de referencia del repositorio.
> - Una notebook en Colab a la que los docentes no tengan permiso de lectura.

Recomendaciones si usan Colab:

- Verifiquen el link desde una ventana de incógnito o con una cuenta ajena al grupo.
- No modifiquen la notebook de Colab después de la fecha de entrega. Como resguardo, también pueden descargarla (*Archivo → Descargar → .ipynb*) y commitearla en el repositorio.
- Guarden los datos y los modelos en Google Drive y documenten en la notebook cómo montarlos y de dónde descargarlos, para que sea reproducible.

### 8.3 Descripción del PR

Usen esta plantilla:

```markdown
## Integrantes
- Nombre Apellido — legajo — @usuario-github
- Nombre Apellido — legajo — @usuario-github

## Resumen
Breve descripción del trabajo realizado y del pipeline implementado.

## Dataset
Splits usados, reglas de filtrado (imágenes descartadas por regla) y conjunto
independiente (cantidad de imágenes, razas, fuente y link de descarga).

## Etapa 1 — Búsqueda por similitud
| Embeddings | Dim | Métrica | NDCG@10 (test) | NDCG@10 (independiente) | Accuracy raza |
|---|---|---|---|---|---|
| baseline (<modelo>) | | | | | |
| ResNet fine-tuned | | | | | |
| CNN propia | | | | | |

**Embeddings elegidos (`EMBEDDING_MODEL`):** ... — **Umbral:** ... (cosine / l2)

## Etapa 2 — Modelos comparados
| Métrica | Modelo A (ResNet fine-tuned) | Modelo B (CNN propia) |
|---|---|---|
| Accuracy (test) | | |
| Precision / Recall / F1 (macro) | | |
| Specificity (macro) | | |
| Accuracy (conjunto independiente) | | |
| Parámetros / Tamaño (MB) | | |
| Tiempo de inferencia CPU (ms/imagen) | | |

**Modelo elegido para clasificar:** ... — **Justificación:** ...

## Etapa 3 — Detección
Detector usado, `YOLO_CONF_THRESHOLD` elegido y capturas (un perro, varios perros,
escena compleja, sin perros).

## Modelos
Links públicos de solo lectura: <links>
Ubicación esperada: `models/<RESNET18_MODEL_NAME>`, `models/<CNN_CUSTOM_MODEL_NAME>`

## Modificaciones fuera de las funciones indicadas
Qué se cambió y por qué (o "ninguna").

## Problemas encontrados y soluciones

## Cómo correrlo
Pasos exactos para reproducir: descargar el dataset y los modelos, configurar
`.env`, indexar la base y `docker compose up`.
```

### 8.4 Checklist antes de entregar

- [ ] El PR apunta a `main` **de su repositorio**, **no está mergeado** y el link está subido al campus.
- [ ] Si el repo es privado, los docentes tienen acceso (Maintain / colaborador).
- [ ] Las notebooks están **ejecutadas y con las salidas visibles**, en GitHub o en Colab.
- [ ] Si usaron Colab: las notebooks de referencia del repo tienen el link y los docentes pueden abrirlo (probado en incógnito).
- [ ] Los links a los modelos y al conjunto independiente funcionan desde una ventana de incógnito.
- [ ] Partiendo de un clon limpio y siguiendo **solo** las instrucciones del PR, `docker compose build` y `docker compose up` levantan sin errores, `/health` responde `ok` y encuentra los checkpoints.
- [ ] Las tres pestañas del frontend funcionan con los modelos del grupo.
- [ ] No quedan `NotImplementedError` en los servicios.

---

## 9. Buenas prácticas y documentación

### 9.1 Buenas prácticas

Se valora especialmente:

- Código modular y separación por capas (servicios, modelos, preprocesamiento, almacenamiento).
- Manejo de errores, con mensajes claros que lleguen a `/status`.
- Logging con el `logger` de cada módulo (no `print`).
- Type hints.
- Tests básicos.
- Reproducibilidad de los experimentos: semillas, versiones fijas y splits guardados.

### 9.2 Documentación

| Qué documentar | Dónde |
|---|---|
| Explicación completa del pipeline | `README.md` (sección *Pipeline*, actualizada según su implementación) |
| Dataset, filtrado, conjunto independiente y preprocesamiento | `1-pre-processing.ipynb` + resumen en el `README.md` |
| Justificación de los modelos elegidos, proceso de entrenamiento e hiperparámetros | `2-train.ipynb` + resumen en el `README.md` |
| Resultados obtenidos y comparación entre enfoques | `2-train.ipynb` + tablas en el PR |
| Problemas encontrados y soluciones implementadas | `README.md` + PR |
| Modificaciones fuera de las funciones indicadas | PR |

---

## 10. Criterios de evaluación

Se evalúan los siguientes aspectos:

- **Parte 1 — Dataset:** EDA, filtrado, conjunto independiente, preprocesamiento, augmentation y splits.
- **Etapa 1:** implementación de la búsqueda, NDCG@10 y su justificación.
- **Etapa 2:** modelos A y B, evaluación rigurosa, comparación, análisis de errores y conclusión.
- **Etapa 3:** detección y clasificación con uno o varios perros y en escenas complejas.
- **Funcionalidad** end-to-end con `docker compose`.
- **Calidad del código y arquitectura:** modularidad, manejo de errores, logging, type hints y tests.
- **Documentación y justificación técnica** de cada decisión, y resultados experimentales.
- **Participación:** historial de commits de ambos integrantes y PR.
- **Defensa oral.**

Condiciones para aprobar:

- El sistema debe funcionar sin errores con `docker compose build` y `docker compose up`, con las tres etapas usando los modelos del grupo.
- Ambas notebooks deben estar ejecutadas y documentadas.
- El PR debe estar abierto, sin mergear y accesible para los docentes.

---

## 11. Recomendaciones generales y errores frecuentes

**Organización**

- Empiecen por el pipeline completo con algo simple (Etapa 1 con el baseline, YOLO y un fine-tuning corto) y después mejoren cada parte. Tener la app funcionando temprano les da un entorno para probar.
- Muevan el código reutilizable (preprocesamiento, arquitectura, carga de modelos) a módulos `.py` e impórtenlos desde las notebooks y desde el backend.
- Fijen semillas (`random`, `numpy`, `torch`) para que los resultados sean reproducibles.
- Visualicen embeddings (PCA o t-SNE), analicen falsos positivos y falsos negativos, evalúen imágenes fuera del dataset y midan todas las métricas que reportan.
- No copien implementaciones sin comprenderlas. Si usan IA, que sea de forma moderada y razonable: en la defensa oral tienen que poder explicar cada decisión y cada línea entregada.

**Errores frecuentes**

- **BGR vs. RGB:** OpenCV lee en BGR; PIL, torchvision y los modelos preentrenados esperan RGB.
- **Normalización distinta** entre entrenamiento e inferencia.
- **Indexar las imágenes de consulta** en la galería: el NDCG@10 sale artificialmente alto.
- **Augmentation sobre validación, test o el conjunto independiente.**
- **Umbral elegido a ojo** o calibrado sobre el mismo conjunto en el que se reportan las métricas.
- **`torch.save(model.state_dict())`** cuando el backend espera el modelo completo, o clase no importable al hacer `torch.load`.
- **Perder el mapeo índice → raza** (por ejemplo, un orden de clases distinto entre la notebook y la app).
- **Dimensión del embedding distinta de `EMBEDDING_DIM`**: pgvector rechaza el insert.
- **Buscar con un modelo distinto al usado para indexar** la base.
- **Cargar el modelo o el detector en cada request:** la app se vuelve muy lenta.
- **Cajas en coordenadas normalizadas** en lugar de píxeles, o fuera de la imagen.
- **Subir datasets o pesos al repositorio.**

**Material de la cátedra relacionado**

- Clase 1 — CNNs desde cero y ResNet50: [Colab](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/Class_1_Convolutions_Datasets_CNN_and_Base_Architecture.ipynb)
- Clase 2 — Arquitecturas modernas, YOLO y *transfer learning*: [Colab](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/Class_2_Modern_CNN.ipynb)
- Clase 2 (lab) — Clasificador con `timm` y cambio de dominio: [Colab](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/Class_2_Hands_on_lab.ipynb)
- Clase 3 — NMS y métricas de evaluación: [Colab](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/class_3_Key_Concepts_for_Deep_Learning_CV.ipynb)
- Clase 4 — Bases de datos vectoriales y embeddings: [Colab](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/class_4_ViT_and_FD.ipynb)
- Clase 7 — YOLO11 y latencia en CPU vs. GPU: [Colab](https://colab.research.google.com/github/EzequielMatiasArevalo/tuia-computer-vision/blob/main/colab/class_7_CPU_vs_GPU_YOLOV11_vs_SAM3.ipynb)

**Recursos útiles**

- 70 Dog Breeds Image Dataset: https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set
- ResNet (He et al., 2015): https://arxiv.org/abs/1512.03385
- Modelos preentrenados de torchvision: https://pytorch.org/vision/stable/models.html
- timm: https://huggingface.co/docs/timm
- Ultralytics YOLO: https://docs.ultralytics.com/
- pgvector: https://github.com/pgvector/pgvector
- Albumentations: https://albumentations.ai/docs/
- Exportar a ONNX: https://pytorch.org/docs/stable/onnx.html
- NDCG (Järvelin y Kekäläinen, 2002): https://doi.org/10.1145/582415.582418

Ante dudas, consulten por los canales de la cátedra. **¡Éxitos!**
