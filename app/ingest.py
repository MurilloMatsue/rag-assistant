# app/ingest.py
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import DOCS_DIR, VECTORSTORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


def carregar_documentos(pasta: str):
    """Carrega todos os PDFs da pasta docs/"""
    loader = PyPDFDirectoryLoader(pasta)
    documentos = loader.load()
    print(f"{len(documentos)} páginas carregadas de {pasta}")
    return documentos


def dividir_chunks(documentos):
    """
    RecursiveCharacterTextSplitter tenta dividir por parágrafos,
    depois por frases, depois por palavras — preserva contexto
    semântico melhor que dividir por número fixo de caracteres.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documentos)
    print(f"{len(chunks)} chunks gerados")
    return chunks


def criar_embeddings():
    """
    all-MiniLM-L6-v2: modelo leve (80MB), roda na CPU,
    produz vetores de 384 dimensões. Ótimo para portfólio.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def indexar_e_salvar(chunks, embeddings):
    """Gera os vetores e salva o índice FAISS em disco"""
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"Índice salvo em {VECTORSTORE_DIR}")
    return vectorstore


def run_ingestion():
    documentos = carregar_documentos(DOCS_DIR)
    chunks     = dividir_chunks(documentos)
    embeddings = criar_embeddings()
    indexar_e_salvar(chunks, embeddings)
    print("Indexação concluída.")


if __name__ == "__main__":
    run_ingestion()