# app/retriever.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import VECTORSTORE_DIR, EMBEDDING_MODEL, TOP_K


def carregar_vectorstore():
    embeddings   = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore  = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True  # necessário no LangChain >= 0.2
    )
    return vectorstore


def buscar_chunks(pergunta: str, vectorstore) -> list:
    """
    similarity_search_with_score retorna os TOP_K chunks
    mais similares à pergunta, junto com o score de distância.
    Score menor = mais próximo = mais relevante.
    """
    resultados = vectorstore.similarity_search_with_score(pergunta, k=TOP_K)
    return resultados