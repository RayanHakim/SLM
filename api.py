import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Menghubungkan ke Qdrant database lokal dan menginisialisasi model...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    client = QdrantClient(path="./qdrant_db")
    
    db = QdrantVectorStore(
        client=client, 
        collection_name="jurnal_koleksi", 
        embedding=embeddings
    )
    
    llm = OllamaLLM(model="phi3", temperature=0.1)
    
    app_state["retriever"] = db.as_retriever(search_kwargs={"k": 5})
    app_state["llm"] = llm
    app_state["client"] = client
    
    print("Sistem RAG dan database Qdrant siap digunakan!")
    yield
    print("Menutup koneksi database Qdrant secara bersih...")
    if "client" in app_state:
        app_state["client"].close()

app = FastAPI(title="Corporate RAG API Backend", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. PROMPT UNTUK MODE DOKUMEN LOKAL (RAG) - Ketat pada data
rag_system_prompt = (
    "Anda adalah analis profesional yang jujur dan ringkas.\n"
    "Tugas Anda adalah menjawab pertanyaan berdasarkan konteks dokumen yang diberikan.\n"
    "Aturan ketat:\n"
    "1. Jawablah pertanyaan secara akurat jika informasinya ada di dalam konteks.\n"
    "2. Jika informasi tentang pertanyaan tersebut tidak disinggung sama sekali di dalam konteks, cukup jawab: "
    "'Maaf, saya tidak menemukan informasi tersebut di dalam dokumen referensi yang tersedia saat ini.'\n"
    "3. Jangan mengulang-ulang kalimat jawaban Anda.\n\n"
    "Konteks:\n{context}\n\n"
    "Pertanyaan: {input}\n"
    "Jawaban Analis:"
)
rag_prompt = ChatPromptTemplate.from_template(rag_system_prompt)

# 2. PROMPT UNTUK MODE UMUM - Bebas menggunakan otak bawaan Phi-3
general_system_prompt = (
    "Anda adalah asisten AI yang cerdas, ramah, dan profesional.\n"
    "Jawablah pertanyaan pengguna dengan pengetahuan umum yang Anda miliki secara jelas, terstruktur, dan akurat.\n\n"
    "Pertanyaan: {input}\n"
    "Jawaban Asisten:"
)
general_prompt = ChatPromptTemplate.from_template(general_system_prompt)

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# Update skema data input agar menerima parameter mode (default ke 'rag')
class ChatQuery(BaseModel):
    question: str
    mode: str = "rag"  # Pilihan: "rag" atau "general"

@app.post("/api/chat")
async def chat_endpoint(query: ChatQuery):
    try:
        llm = app_state.get("llm")
        if not llm:
            raise HTTPException(status_code=500, detail="Aplikasi belum siap.")
            
        # ==================== SKENARIO MODE UMUM ====================
        if query.mode == "general":
            # Langsung satukan prompt umum dengan LLM tanpa lewat Qdrant
            chain = general_prompt | llm | StrOutputParser()
            answer = chain.invoke({"input": query.question})
            
            return {
                "status": "success",
                "answer": answer,
                "sources": []  # Mode umum tidak memiliki referensi dokumen
            }
            
        # ==================== SKENARIO MODE DOKUMEN LOKAL (RAG) ====================
        else:
            retriever = app_state.get("retriever")
            if not retriever:
                raise HTTPException(status_code=500, detail="Retriever belum siap.")
                
            docs = retriever.invoke(query.question)
            if not docs:
                return {
                    "status": "success",
                    "answer": "Maaf, tidak ada data dokumen yang relevan.",
                    "sources": []
                }
                
            context_text = format_docs(docs)
            rag_chain = rag_prompt | llm | StrOutputParser()
            
            input_data = {"context": context_text, "input": query.question}
            answer = rag_chain.invoke(input_data)
            
            sources = []
            for doc in docs:
                file_title = doc.metadata.get('title') or doc.metadata.get('source') or 'Dokumen Lokal'
                sources.append({
                    "file": file_title,
                    "page": doc.metadata.get('page', 1),
                    "content": doc.page_content
                })
                
            return {
                "status": "success",
                "answer": answer,
                "sources": sources
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan internal: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<html><body><h2>File 'index.html' tidak ditemukan!</h2></body></html>"