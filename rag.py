from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

try:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=env_path, override=True)

    # 1. Učitaj MD
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Aleksa_Ljujic_CV.md")
    loader = TextLoader(file_path, encoding="utf-8")
    md_text = loader.load()[0].page_content

    print("Uspešno učitan MD fajl")

    # 2. Split po markdown headerima PRVO
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("##", "h2"),  # Sekacije tipa [CONTACT], [EDUCATION], itd.
        ]
    )
    header_chunks = header_splitter.split_text(md_text)

    # 3. Dodatno split svakog header chunka na manje delove
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Prilagodi po potrebi
        chunk_overlap=50
    )
    
    final_chunks = []
    for chunk in header_chunks:
        # Split svaki header chunk na manje delove
        sub_chunks = text_splitter.split_text(chunk.page_content)
        for sub_chunk in sub_chunks:
            final_chunks.append(
                Document(
                    page_content=sub_chunk,
                    metadata=chunk.metadata  # Zadrži header metadata
                )
            )

    print(f"Broj finalnih chunkova: {len(final_chunks)}")

    # 4. Embeddings + Chroma
    persist_dir = "db"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=final_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print("Vektorska baza sačuvana!")

except FileNotFoundError as e:
    print(f"Greška: Fajl nije pronađen → {e}")
except ValueError as e:
    print(f"Greška: Problem sa podacima → {e}")
except Exception as e:
    print(f"Neočekivana greška: {e}")