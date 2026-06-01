import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL, GROQ_API_KEY

TEMPLATE = """
Você é um assistente especializado em responder perguntas
baseado exclusivamente nos documentos fornecidos.

Contexto recuperado dos documentos:
{contexto}

Pergunta do usuário:
{pergunta}

Instruções:
- Responda apenas com base no contexto acima.
- Se a resposta não estiver no contexto, diga explicitamente
  que a informação não foi encontrada nos documentos.
- Cite de qual parte do documento a resposta veio.
- Seja claro e objetivo.

Resposta:
"""

def criar_chain(vectorstore):
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2
    )

    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=TEMPLATE
    )

    def formatar_contexto(docs_com_score):
        partes = []
        for doc, score in docs_com_score:
            fonte = os.path.basename(doc.metadata.get("source", "desconhecida"))
            pagina = doc.metadata.get("page", "?")
            partes.append(
                f"[Fonte: {fonte} | Página: {pagina} | Score: {score:.3f}]\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(partes)

    def responder(pergunta: str, vectorstore) -> dict:
        from retriever import buscar_chunks
        chunks_encontrados = buscar_chunks(pergunta, vectorstore)
        contexto           = formatar_contexto(chunks_encontrados)

        chain    = prompt | llm | StrOutputParser()
        resposta = chain.invoke({
            "contexto": contexto,
            "pergunta": pergunta
        })
        return {
            "resposta": resposta,
            "fontes":   chunks_encontrados,
            "contexto": contexto
        }

    return responder