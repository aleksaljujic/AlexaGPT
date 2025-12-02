from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from chromadb.config import Settings
from dotenv import load_dotenv
import os

try:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=env_path, override=True)

    # 1. Učitaj MD
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Aleksa_Ljujic_CV.md")
    loader = TextLoader(file_path, encoding="utf-8")
    md_text = loader.load()[0].page_content

    print("Uspesno ucitan MD fajl")

    # 2. Split po markdown headerima (idealno za sekcije/projekte)
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("##", "h2"),
        ]
    )


    #splits = markdown_header_splitter.split_text(md_text)
    chunks = splitter.split_text(md_text)

    print(f"Broj chunkova: {len(chunks)}")

    # 3. Embeddings + Chroma (NOVI API)
    persist_dir = "db"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir  # ovo je dovoljno
    )
    print("Vektorska baza sačuvana!")

except FileNotFoundError as e:
    print(f"Greška: PDF fajl nije pronađen → {e}")

except ValueError as e:
    print(f"Greška: Problem sa podacima, chunk-ovima ili embeddingom → {e}")

except Exception as e:
    print(f"Neočekivana greška: {e}")



