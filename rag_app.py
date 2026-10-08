"""
Simple RAG app over the PMI-ACP Glossary PDF using LangChain + FAISS + OpenAI.

How it works:
  1. First run: reads the PDF, splits it into chunks, creates embeddings locally,
     and saves a FAISS index to the local folder "faiss_index".
  2. Later runs: loads the saved FAISS index from disk (no re-indexing).
  3. Every question retrieves chunks from FAISS, and OpenAI writes the answer
     using only those chunks.

Setup (one time):
  python3 -m pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-openai faiss-cpu sentence-transformers pypdf
  Set your OpenAI key in the terminal:  export OPENAI_API_KEY="sk-..."
  Put the PDF in the same folder as this file, named PMIACP-Glossary.pdf

Run:
  python rag_app.py

Tip: if you change the PDF, delete the "faiss_index" folder so it rebuilds.
"""

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------- Settings ----------------
PDF_PATH = "PMIACP-Glossary.pdf"                          # your PDF file
INDEX_DIR = "faiss_index"                                 # local folder for the FAISS index
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"    # free, runs locally
LLM_MODEL = "gpt-4o-mini"                                 # any OpenAI chat model your key can use
TOP_K = 4                                                 # chunks retrieved per question


# ---------------- Step 1: Build or load the FAISS vector DB ----------------
def build_or_load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Reuse the saved index if it exists
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print("Loading FAISS index from disk...")
        return FAISS.load_local(
            INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,  # safe: we created this file ourselves
        )

    print("Building FAISS index from PDF (first run only)...")
    pages = PyPDFLoader(PDF_PATH).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks")

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(INDEX_DIR)
    print(f"Saved FAISS index to '{INDEX_DIR}/'")
    return vector_db


# ---------------- Step 2: Prompt + LLM ----------------
PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful PMI-ACP exam tutor.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the glossary."

Context:
{context}

Question: {question}

Answer:"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------- Step 3: Ask questions ----------------
def main():
    if not os.getenv("OPENAI_API_KEY"):
        print('OPENAI_API_KEY is not set. Run:  export OPENAI_API_KEY="sk-..."')
        return

    vector_db = build_or_load_vector_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    chain = PROMPT | llm | StrOutputParser()

    print("\nPMI-ACP Glossary RAG ready. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        # 1) Retrieve relevant chunks from FAISS
        docs = retriever.invoke(question)

        # 2) Generate the answer using only those chunks
        answer = chain.invoke({"context": format_docs(docs), "question": question})

        pages = sorted({doc.metadata.get("page", 0) + 1 for doc in docs})
        print(f"\nAnswer: {answer}")
        print(f"(Sources: page {', '.join(map(str, pages))})\n")


if __name__ == "__main__":
    main()