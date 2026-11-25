import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("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")

# 2. Document Loading
pdf_path = os.path.join(os.path.dirname(__file__), "document_loader", "STATS.pdf")
print(f"Loading document from: {pdf_path}")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print(f"Loaded {len(docs)} document pages.")

# 3. Document Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} text chunks.")

# 4. Embeddings & In-Memory Vector Store (Qdrant + FastEmbed)
print("Creating embeddings and indexing into Qdrant vector store...")
embeddings = FastEmbedEmbeddings()

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="rag_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="rag_collection",
    embedding=embeddings,
)
vectorstore.add_documents(chunks)
print("Indexing complete!")

# 5. Initialize NVIDIA LLM
# Note: nemotron uses reasoning tokens, so max_completion_tokens should be >= 1024
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    nvidia_api_key=api_key,
    temperature=0.2,
    max_completion_tokens=2500,
)

# 6. Retriever & RAG Prompt
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert tutor in Statistical Techniques. "
        "Answer the question accurately based ONLY on the provided context. "
        "If the answer cannot be found in the context, say that you don't know.\n\n"
        "Context:\n{context}"
    ),
    ("human", "{question}"),
])

rag_chain = prompt | llm | StrOutputParser()

# 7. Test RAG with a question
def query_rag(question: str):
    print("\n" + "="*60)
    print(f"Query: {question}")
    print("="*60)
    
    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    print(f"\nRetrieved {len(retrieved_docs)} relevant context chunk(s):")
    for i, doc in enumerate(retrieved_docs, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        preview = doc.page_content.strip().replace("\n", " ")[:150]
        print(f" [{i}] (Page {page_num}): {preview}...")

    # Build context string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Generate response
    print("\nGenerating response from NVIDIA LLM...")
    response = rag_chain.invoke({"context": context, "question": question})
    print("\nAnswer:")
    print(response)
    print("="*60)
    return response

if __name__ == "__main__":
    test_question = "What are the normal equations for fitting a straight line using the principle of least squares?"
    query_rag(test_question)