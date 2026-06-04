---
title: RAG Assistant
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---
# 🔍 RAG Assistant — Pergunte sobre seus documentos

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-orange?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Interface-red?style=flat-square)

Um assistente inteligente que permite conversar com qualquer documento PDF em linguagem natural, usando **Retrieval-Augmented Generation (RAG)** para responder perguntas com base no conteúdo real dos arquivos — sem alucinações, com citação de fontes.

---

## 📌 O problema que este projeto resolve

LLMs como GPT e LLaMA não conhecem documentos privados e têm conhecimento estático. Fine-tuning é caro e lento. **RAG resolve isso**: em vez de retreinar o modelo, o sistema busca os trechos mais relevantes do documento em tempo real e os injeta no contexto da pergunta — o modelo responde com base em fatos reais.

**Casos de uso reais:**
- Assistente jurídico sobre contratos
- Suporte técnico sobre manuais de produto
- Chatbot de RH sobre políticas internas
- Pesquisa sobre papers científicos
- Atendimento ao cliente com base em documentação

---

## 🏗️ Arquitetura

O sistema opera em dois pipelines distintos:

```
PIPELINE DE INDEXAÇÃO (roda uma vez)
─────────────────────────────────────────────────────────
PDF → PyPDFLoader → RecursiveCharacterTextSplitter
    → HuggingFace Embeddings → FAISS (salvo em disco)

PIPELINE DE CONSULTA (tempo real)
─────────────────────────────────────────────────────────
Pergunta → Embedding → Busca FAISS (top-k chunks)
         → Montagem do prompt → Groq LLaMA 3.1
         → Resposta com fontes citadas
```

---

## 🛠️ Stack técnica

| Componente | Tecnologia | Por quê |
|---|---|---|
| Orquestração | LangChain | Modularidade — cada componente é substituível independentemente |
| Vector Store | FAISS | Busca de similaridade otimizada, roda localmente sem custo |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) | Leve (80MB), roda na CPU, excelente custo-benefício |
| LLM | Groq + LLaMA 3.1 8B | API gratuita, baixa latência, alta qualidade |
| Interface | Streamlit | Permite demo interativa com upload e histórico em poucas linhas |
| Chunking | RecursiveCharacterTextSplitter | Preserva coerência semântica respeitando parágrafos e frases |

---

## 🔑 Decisões técnicas

**Chunk overlap de 50 tokens**
Informações relevantes podem estar na fronteira entre dois chunks. Com overlap, cada fronteira aparece em ambos os vizinhos — nenhum contexto se perde na divisão.

**Busca semântica vs keyword**
A busca vetorial encontra "animal doméstico" quando o usuário pergunta "cachorro". Fundamental para documentos técnicos onde o usuário não sabe o jargão exato do documento.

**Score de similaridade exposto na UI**
O usuário consegue avaliar a confiança da resposta. Endereça diretamente o problema de alucinação — se o score for baixo, a resposta pode não ser confiável.

**Arquitetura modular**
Cada componente é isolado (`ingest.py`, `retriever.py`, `chain.py`). Trocar FAISS por Pinecone ou Groq por GPT-4 é uma mudança de uma linha no `config.py`.

---

## 🚀 Como rodar localmente

### Pré-requisitos
- Python 3.10+
- Chave de API do Groq (gratuita em [console.groq.com](https://console.groq.com))

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/rag-assistant.git
cd rag-assistant

# Crie o ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt
```

### Configuração

Crie um arquivo `.env` na raiz do projeto:

```
GROQ_API_KEY=sua_chave_aqui
```

### Execução

```bash
streamlit run app/main.py
```

Acesse `http://localhost:8501`, faça upload de um PDF e comece a perguntar.

---

## 📁 Estrutura do projeto

```
rag-assistant/
├── app/
│   ├── main.py          # Interface Streamlit
│   ├── ingest.py        # Pipeline de indexação
│   ├── retriever.py     # Busca vetorial no FAISS
│   ├── chain.py         # Montagem do prompt e chamada ao LLM
│   └── config.py        # Constantes e configurações
├── docs/                # PDFs para indexar (não versionado)
├── vectorstore/         # Índice FAISS gerado (não versionado)
├── .env                 # Chaves de API (não versionado)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔄 Como escalar para produção

Este projeto usa FAISS local e Groq free tier — ideal para portfólio e MVPs. Para produção:

| Componente | Dev (este projeto) | Produção |
|---|---|---|
| Vector Store | FAISS local | Pinecone / Weaviate |
| LLM | Groq free tier | GPT-4 / Claude / Gemini |
| Backend | Streamlit | FastAPI + React |
| Ingestão | Síncrona | Celery + Kafka |
| Monitoramento | — | Evidently AI + Prometheus |

---

## 📚 Conceitos aplicados

- **RAG** (Retrieval-Augmented Generation)
- **Embeddings** e espaço vetorial semântico
- **Busca por similaridade** com distância cosseno
- **Chunking** com overlap para preservação de contexto
- **Prompt engineering** para respostas factuais
- **Pipeline de ML** modular e desacoplado

---

## 👤 Autor

Desenvolvido por **Murillo Matsue** como parte de um portfólio em Engenharia de IA.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-seu--perfil-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/murilloichiro)
[![GitHub](https://img.shields.io/badge/GitHub-seu--usuario-black?style=flat-square&logo=github)](https://github.com/MurilloMatsue)