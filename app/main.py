import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from retriever import carregar_vectorstore
from chain import criar_chain
from ingest import carregar_documentos, dividir_chunks, criar_embeddings, indexar_e_salvar
from config import DOCS_DIR, VECTORSTORE_DIR

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide"
)

st.title("RAG Assistant — pergunte sobre seus documentos")
st.caption("Faça upload de PDFs e pergunte qualquer coisa sobre eles.")

# --- Upload de PDFs ---
uploaded_files = st.file_uploader(
    "Envie seus PDFs aqui",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs(DOCS_DIR, exist_ok=True)
    novos_arquivos = []

    for file in uploaded_files:
        destino = os.path.join(DOCS_DIR, file.name)
        if not os.path.exists(destino):
            with open(destino, "wb") as f:
                f.write(file.read())
            novos_arquivos.append(file.name)

    if novos_arquivos:
        with st.spinner(f"Indexando {len(novos_arquivos)} novo(s) arquivo(s)..."):
            documentos = carregar_documentos(DOCS_DIR)
            chunks     = dividir_chunks(documentos)
            embeddings = criar_embeddings()
            indexar_e_salvar(chunks, embeddings)
            st.cache_resource.clear()
        st.success(f"{len(novos_arquivos)} arquivo(s) indexado(s): {', '.join(novos_arquivos)}")
    else:
        st.info("Arquivos já indexados.")

# --- Carrega o sistema ---
@st.cache_resource
def carregar_sistema():
    vectorstore = carregar_vectorstore()
    responder   = criar_chain(vectorstore)
    return vectorstore, responder

try:
    vectorstore, responder = carregar_sistema()
    st.success("Índice carregado. Sistema pronto.")
except Exception as e:
    st.warning("Nenhum documento indexado ainda. Faça upload de um PDF acima.")
    st.stop()

# --- Histórico ---
if "historico" not in st.session_state:
    st.session_state.historico = []

# --- Input ---
pergunta = st.chat_input("Digite sua pergunta sobre os documentos...")

if pergunta:
    st.session_state.historico.append({"role": "user", "content": pergunta})

    with st.spinner("Buscando nos documentos..."):
        resultado = responder(pergunta, vectorstore)

    st.session_state.historico.append({
        "role":    "assistant",
        "content": resultado["resposta"],
        "fontes":  resultado["fontes"]
    })

# --- Renderiza histórico ---
for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant" and "fontes" in msg:
            with st.expander("Ver fontes utilizadas"):
                for doc, score in msg["fontes"]:
                    fonte  = os.path.basename(doc.metadata.get("source", "?"))
                    pagina = doc.metadata.get("page", "?")
                    st.markdown(f"**Score:** `{score:.4f}` | **Arquivo:** `{fonte}` | **Página:** `{pagina}`")
                    st.text(doc.page_content[:300] + "...")
                    st.divider()