from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="Corporate RAG API Backend", version="1.0")

# Setup CORS biar bisa ditembak dari index.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load database vektor dan model Phi-3 lokal
print("Menghubungkan ke ChromaDB dan Ollama Phi-3...")
embeddings = OllamaEmbeddings(model="phi3")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="phi3", temperature=0.2)

# Prompt template untuk batasi jawaban AI sesuai dokumen
system_prompt = (
    "Anda adalah analis keuangan profesional yang jujur. "
    "Jawablah pertanyaan pengguna hanya berdasarkan konteks dokumen yang diberikan di bawah ini.\n\n"
    "Konteks: {context}\n\n"
    "Pertanyaan: {input}\n"
    "Jawaban:"
)
prompt = ChatPromptTemplate.from_template(system_prompt)

# Fungsi helper buat gabungin potongan dokumen (stuffing)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Validasi data masuk
class ChatQuery(BaseModel):
    question: str

# Endpoint utama untuk chat RAG
@app.post("/api/chat")
async def chat_endpoint(query: ChatQuery):
    try:
        # 1. Ambil dokumen relevan dari database secara manual dulu
        docs = retriever.invoke(query.question)
        context_text = format_docs(docs)
        
        # 2. Rangkai input untuk prompt template
        input_data = {"context": context_text, "input": query.question}
        
        # 3. Jalankan pipeline LCEL murni (Prompt | LLM | Parser)
        rag_chain = prompt | llm | StrOutputParser()
        answer = rag_chain.invoke(input_data)
        
        # 4. Ekstrak info file dan halaman asli buat sitasi data
        sources = []
        for doc in docs:
            sources.append({
                "file": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'Unknown'),
                "content": doc.page_content
            })
            
        return {
            "status": "success",
            "answer": answer,
            "sources": sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Cek status server
@app.get("/")
def index():
    return {"message": "Server RAG lokal aktif, Le!"}