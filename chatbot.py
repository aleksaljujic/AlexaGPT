from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

try:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=env_path, override=True)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

    llm = ChatOpenAI(model="gpt-4o-mini")

except Exception as e:
    print("Greška u inicijalizaciji sistema:", e)
    raise 



def rag(question: str) -> str:
    try:
        # Prompt template
        prompt = ChatPromptTemplate.from_template(
            """
            Koristi isključivo sledeći kontekst da odgovoriš precizno, tačno i potpuno na pitanja o Aleksi Ljujić.
            Nikada ne koristi informacije koje nisu eksplicitno prisutne u kontekstu.

            Jezik odgovora

            Ako je pitanje postavljeno na srpskom jeziku, odgovori na srpskom.

            Ako je pitanje postavljeno na engleskom jeziku, odgovori na engleskom.

            Pravila razumevanja pitanja (INTENTI)

            Ako je pitanje opšte (npr. „Ko je Aleksa?“, „Tell me about Aleksa“, „Who is Aleksa?“),
            napravi kratak sažetak koristeći [ABOUT ME] sekciju i osnovne informacije iz CV-a.

            Ako se pita o projektima, izlistaj sve projekte iz sekcije [PROJECTS], sa njihovim nazivima, tehnologijama i kratkim opisima.

            Ako se pita o radnom ili industrijskom iskustvu, koristi isključivo informacije iz sekcije
            [EXPERIENCE] – Integration Architect Intern — STADA GIS Serbia.

            Ako se pita o vannastavnim, studentskim ili volonterskim aktivnostima, koristi isključivo sekciju
            [STUDENT ORGANIZATIONS].

            Ako se pita o obrazovanju, koristi isključivo sekciju [EDUCATION].

            Ako se pita o kontakt informacijama, koristi isključivo sekciju [CONTACT].

            Ako se pita o veštinama, koristi isključivo sekciju [SKILLS].

            Ako se pita o projektima koristi iskljucivo sekciju [PROJECTS]

            Ako se pita o iskustvu korisit iskljucivo sekcije [PROJECTS] i [EXPERIENCE] 

            Nedostajuće informacije (STRICT FALLBACK)

            Ako tražena informacija ne postoji u kontekstu, odgovori tačno:

            „Ta informacija nije u dokumentu.“ (za srpski)

            „That information is not in the document.“ (za engleski)

            Dodatna pravila

            Ne dodaj pretpostavke, lična mišljenja, interpretacije ili informacije van konteksta.

            Ne spajaj informacije ako to nije eksplicitno dozvoljeno pravilima iznad.

            Odgovor mora biti jasan, koncizan i fokusiran isključivo na činjenice iz CV-a.

            Uzmi u obzir da je praksa bila 2024 a danas je 2026.

            KONTEKST:

            {context}

            PITANJE:

            {question}

            ODGOVOR:
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

        print("===== CONTEXT START =====")
        print(context_text)
        print("===== CONTEXT END =====")

        return answer

    except Exception as e:
        return f"Došlo je do greške u RAG procesu: {str(e)}"

#print(rag("Na kojim projektima je aleksa radio"))
#print(rag("Ko je Aleksa"))

