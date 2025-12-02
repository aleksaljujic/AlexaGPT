from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os


# -----------------------------
# INITIALIZATION (global)
# -----------------------------
try:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=env_path, override=True)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    llm = ChatOpenAI(model="gpt-4o-mini")

except Exception as e:
    print("Greška u inicijalizaciji sistema:", e)
    raise  # bitno – aplikacija ne treba da radi bez ispravnog RAG-a


# -----------------------------
# RAG FUNCTION – poziva frontend
# -----------------------------
def rag(question: str) -> str:
    try:
        # Prompt template
        prompt = ChatPromptTemplate.from_template(
            """
            Koristi isključivo sledeći kontekst da odgovoriš precizno i potpuno na pitanja o Aleksi Ljujić.
            Ako je pitanje postavljeno na primer koje projekte ili iskustvo ima zelim da izlistas sve sto je u sekciji RELEVANT PROJECTS. Ako se pita za vannastavne i volonterske aktivnosti onda sta je radio u sekciji STUDENT ORGANIZATIONS. Ako se pita za iskustvo iz industrije ili radno iskustvo onda INTERNSHIP u STADA GIS.
            Ako informacija nije u kontekstu, reci "Ta informacija nije u dokumentu."

            KONTEKST:
            {context}

            PITANJE:
            {question}

            Odgovor (jasan, koncizan, fokusiran na relevantne činjenice iz CV-a):
            """
        )


        # Retrieving relevant docs
        docs = retriever.invoke(question)
        context_text = "\n\n".join(d.page_content for d in docs)

        # Build chain
        chain = prompt | llm | StrOutputParser()

        # Execute chain
        answer = chain.invoke({
            "context": context_text,
            "question": question
        })

        return answer

    except Exception as e:
        return f"Došlo je do greške u RAG procesu: {str(e)}"

print(rag("Na kojim projektima je aleksa radio"))