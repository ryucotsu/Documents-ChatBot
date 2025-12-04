import streamlit as st
import os
from PyPDF2 import PdfReader

# --- Bloque de Importaciones (Versión 2025) ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # Fallback to the langchain subpackage path if available in other envs
    from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import google.generativeai as genai


class GoogleGenAIEmbeddings(Embeddings):
    """Minimal embeddings wrapper compatible with `FAISS.from_texts`.

    Uses `google.generativeai.embed_content` under the hood. The method
    `embed_documents` returns a list of embedding vectors (list[float]).
    """
    def __init__(self, model: str = "models/text-embedding-004", google_api_key: str | None = None):
        self.model = model
        if google_api_key:
            genai.configure(api_key=google_api_key)

    def _parse_response(self, resp, texts_len: int = 1):
        """Normalize various response shapes from `genai.embed_content` into
        a list of vectors `List[List[float]]`.
        """
        out = []
        # Normalize dict-like responses
        if isinstance(resp, dict):
            if 'embeddings' in resp:
                candidate = resp['embeddings']
            elif 'data' in resp and isinstance(resp['data'], list):
                candidate = [item.get('embedding', item) if isinstance(item, dict) else item for item in resp['data']]
            elif 'embedding' in resp:
                candidate = resp['embedding']
            else:
                candidate = resp
        else:
            candidate = resp

        # Candidate may be a single vector, a list of vectors, or list of dicts
        if isinstance(candidate, (list, tuple)):
            for item in candidate:
                vec = None
                if isinstance(item, dict):
                    # common shape: {'embedding': [...]} or other wraps
                    if 'embedding' in item:
                        vec = item['embedding']
                    else:
                        # take first list-like value
                        for v in item.values():
                            if isinstance(v, (list, tuple)):
                                vec = v
                                break
                else:
                    vec = item

                if vec is None:
                    continue

                if isinstance(vec, (list, tuple)):
                    import numpy as _np
                    arr = _np.array(vec, dtype=float)
                    # flatten any extra nesting to 1D
                    if arr.ndim > 1:
                        arr = arr.ravel()
                    out.append(arr.tolist())
                else:
                    # single numeric -> wrap
                    try:
                        out.append([float(vec)])
                    except Exception:
                        continue
        else:
            # single vector-like
            if isinstance(candidate, dict) and 'embedding' in candidate:
                candidate = candidate['embedding']
            if isinstance(candidate, (list, tuple)):
                import numpy as _np
                arr = _np.array(candidate, dtype=float)
                if arr.ndim > 1:
                    arr = arr.ravel()
                out.append(arr.tolist())
            else:
                try:
                    out.append([float(candidate)])
                except Exception:
                    pass

        # If API returned a single embedding for a batch, replicate it (best-effort)
        if len(out) == 1 and texts_len > 1:
            out = [out[0] for _ in range(texts_len)]

        return out

    def embed_documents(self, texts):
        # embed_content accepts an iterable of content
        resp = genai.embed_content(model=self.model, content=texts)
        embeddings = self._parse_response(resp, texts_len=len(texts))
        if not embeddings:
            raise ValueError(f"No embeddings returned by embed_content. Response repr: {repr(resp)[:200]}")
        # ensure consistent 2D shape
        first_len = len(embeddings[0])
        for v in embeddings:
            if len(v) != first_len:
                raise ValueError("Inconsistent embedding dimensionality returned by embed_content")
        return embeddings

    def embed_query(self, text: str):
        out = self.embed_documents([text])
        return out[0] if out else []


# --- Configuración ---
st.set_page_config(page_title="Chat PDF", layout="wide")
st.title("🤖 Chat con Documentos")

# --- Barra Lateral ---
with st.sidebar:
    api_key = st.text_input("Google API Key:", type="password")
    pdf_docs = st.file_uploader("Sube tus PDFs", accept_multiple_files=True, type=['pdf'])
    procesar = st.button("Procesar")

# --- Funciones ---
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text() or ""
        except: pass
    return text

def get_vector_store(text, key):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    embeddings = GoogleGenAIEmbeddings(model="models/text-embedding-004", google_api_key=key)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_answer(key, docs, query):
    # Use google.generativeai directly to generate an answer using the docs as context
    genai.configure(api_key=key)
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    chat = model.start_chat()
    # Build context from retrieved documents
    context = "\n\n---\n\n".join([getattr(d, 'page_content', str(d)) for d in docs])
    prompt = f"Contexto:\n{context}\n\nPregunta:\n{query}\n\nRespuesta:"
    resp = chat.send_message(prompt)
    # Try common response shapes
    if hasattr(resp, 'text'):
        return resp.text
    gens = getattr(resp, 'generations', None)
    if gens:
        first = gens[0]
        return getattr(first, 'text', str(first))
    return str(resp)

# --- Ejecución ---
if procesar and api_key and pdf_docs:
    with st.spinner("Procesando..."):
        text = get_pdf_text(pdf_docs)
        if text:
            get_vector_store(text, api_key)
            st.success("¡Listo!")

q = st.text_input("Pregunta:")
if q and api_key and os.path.exists("faiss_index"):
    try:
        embeddings = GoogleGenAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(q)
        # get an answer directly using the Google GenAI chat session
        res = get_answer(api_key, docs, q)
        st.write(res)
    except Exception as e:
        st.error(f"Error: {e}")