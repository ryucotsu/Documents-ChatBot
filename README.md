# 🤖 Chat con Documentos

Una aplicación Streamlit que permite cargar PDFs, procesarlos con embeddings de Google Generative AI y hacer preguntas sobre su contenido usando un modelo de lenguaje.

## Características

- 📄 **Carga de PDFs**: Sube múltiples archivos PDF simultáneamente
- 🔍 **Búsqueda Semántica**: Extrae chunks del texto y crea embeddings usando Google Generative AI
- 💬 **Chat Inteligente**: Responde preguntas sobre los documentos usando Gemini 2.5 Flash
- 💾 **Persistencia**: Guarda los índices FAISS localmente para reutilización

## Requisitos Previos

- Python 3.10+
- pip (gestor de paquetes de Python)
- Una API Key de Google Generative AI (obtén una en [Google AI Studio](https://aistudio.google.com/app/apikey))

## Instalación

### 1. Clonar o descargar el proyecto

```bash
git clone <repositorio>
cd codespaces-blank
```

### 2. Crear un entorno virtual (opcional pero recomendado)

```bash
python -m venv .venv
```

### 3. Activar el entorno virtual

**En Linux/macOS:**
```bash
source .venv/bin/activate
```

**En Windows:**
```bash
.venv\Scripts\activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Uso

### Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá en tu navegador en `http://localhost:8501`

### Flujo de Uso

1. **Ingresar API Key**: En la barra lateral, introduce tu Google API Key
2. **Cargar PDFs**: Usa el widget de carga para seleccionar uno o más PDFs
3. **Procesar**: Haz clic en "Procesar" para:
   - Extraer texto de los PDFs
   - Dividir el texto en chunks
   - Generar embeddings
   - Crear índice FAISS
4. **Hacer Preguntas**: Escribe una pregunta sobre el contenido de los PDFs
5. **Obtener Respuestas**: El modelo generará una respuesta basada en el contexto extraído

## Estructura del Proyecto

```
codespaces-blank/
├── app.py                 # Aplicación principal Streamlit
├── requirements.txt       # Dependencias de Python
├── README.md             # Este archivo
├── faiss_index/          # Directorio con índices FAISS (creado al procesar)
│   ├── index.faiss
│   └── index.pkl
└── .venv/               # Entorno virtual (generado localmente)
```

## Archivo: requirements.txt

```
streamlit==1.52.0
PyPDF2==3.0.1
langchain==1.1.0
langchain-core==1.1.0
langchain-community==0.4.1
langchain-text-splitters==1.0.0
faiss-cpu==1.13.0
google-generativeai==0.8.5
```

## Archivo: app.py

Contiene:
- **GoogleGenAIEmbeddings**: Clase personalizada que hereda de `langchain_core.embeddings.Embeddings` para generar embeddings usando la API de Google
- **get_pdf_text()**: Extrae texto de archivos PDF
- **get_vector_store()**: Crea un índice FAISS con embeddings
- **get_answer()**: Genera respuestas usando el modelo Gemini
- **Interfaz Streamlit**: Componentes UI para cargar, procesar y consultar

## Características Técnicas

### Embeddings Personalizados
La clase `GoogleGenAIEmbeddings` implementa:
- Parsing flexible de múltiples formatos de respuesta de la API
- Normalización automática de dimensionalidad
- Validación de consistencia de embeddings

### Procesamiento de PDFs
- **Splitter**: RecursiveCharacterTextSplitter con tamaño de chunk de 10,000 caracteres y overlap de 1,000
- **Vectorización**: Embeddings de Google (modelo: `models/text-embedding-004`)
- **Almacenamiento**: FAISS IndexFlatL2 para búsqueda de similitud

### Generación de Respuestas
- Modelo: Gemini 2.5 Flash
- Usa similitud semántica para recuperar contexto relevante
- Construye prompts con documento y pregunta del usuario

## Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'langchain_text_splitters'"
- Asegúrate de estar usando el entorno virtual correcto
- Reinstala las dependencias: `pip install -r requirements.txt`

### Error: "ValueError: too many values to unpack"
- Este error fue resuelto en la implementación actual
- La clase `GoogleGenAIEmbeddings` normaliza automáticamente formatos de respuesta

### Error: "API Key inválida"
- Verifica que tu API Key sea correcta en [Google AI Studio](https://aistudio.google.com/app/apikey)
- Asegúrate de tener acceso habilitado a Google Generative AI

### La app está lenta
- Los embeddings se generan bajo demanda
- Para muchos documentos, el procesamiento inicial puede tardar
- Los índices se guardan en `faiss_index/` para reutilización rápida

## Dependencias Principales

- **Streamlit**: Framework web para aplicaciones de datos
- **LangChain**: Orquestación de LLMs y herramientas de IA
- **FAISS**: Búsqueda de similitud vectorial eficiente
- **PyPDF2**: Extracción de texto desde PDFs
- **Google Generative AI**: API de embeddings y LLM de Google

## Versiones Probadas

- Python 3.12.3
- Las versiones exactas de todas las dependencias están especificadas en `requirements.txt`

## Próximas Mejoras Posibles

- [ ] Soporte para más formatos de documentos (DOCX, TXT, etc.)
- [ ] Historial de conversación persistente
- [ ] Configuración de parámetros de modelo (temperatura, top_k, etc.)
- [ ] Exportar respuestas a PDF
- [ ] Múltiples índices para diferentes categorías de documentos
- [ ] Caché de embeddings para velocidad

## Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## Soporte

Si encuentras problemas o tienes preguntas:
1. Revisa la sección "Solución de Problemas"
2. Verifica que todas las dependencias estén instaladas correctamente
3. Asegúrate de tener una API Key válida de Google
