# Corporate Financial RAG Chatbot using Phi-3 & FastAPI

Proyek ini adalah sistem *Retrieval-Augmented Generation* (RAG) lokal yang dirancang untuk melakukan analisis dokumen keuangan dan laporan tahunan perusahaan (*Annual Report*). Sistem ini mengekstrak data dari dokumen PDF yang tebal, menyimpannya ke dalam database vektor lokal, dan menyediakan antarmuka pencarian berbasis AI yang akurat tanpa membocorkan data ke pihak ketiga (100% lokal).

## 🚀 Fitur Utama
- **Local Execution**: Menggunakan model *Small Language Model* (SLM) Phi-3 via Ollama yang berjalan sepenuhnya di GPU lokal.
- **Traceability / Citation**: Chatbot tidak hanya memberikan jawaban, tetapi juga melampirkan nama file dokumen beserta nomor halaman tempat informasi tersebut ditemukan untuk menghindari halusinasi medis/finansial.
- **Modern LCEL Pipeline**: Menggunakan *LangChain Expression Language* (LCEL) untuk efisiensi rantai proses (*chaining*) dokumen yang stabil.
- **Lightweight Vanilla Frontend**: Antarmuka berbasis HTML5, CSS3, dan Vanilla JavaScript murni tanpa ketergantungan pada framework eksternal.

## 🛠️ Tech Stack
- **Backend Framework**: FastAPI (Python)
- **Orchestration**: LangChain Core / Community
- **LLM & Embeddings**: Phi-3 (Ollama)
- **Vector Database**: ChromaDB
- **PDF Parser**: PDFPlumber
- **Frontend**: HTML, CSS, Vanilla JavaScript (Fetch API)

---

## 📂 Struktur Proyek

```text
SLM/
├── data/                 # Tempat menyimpan berkas PDF laporan keuangan
├── chroma_db/            # Database vektor lokal hasil ekstraksi (Chroma)
├── ingest.py             # Skrip untuk membaca, memotong (chunking), dan embedding PDF
├── api.py                # Server backend FastAPI & pipeline RAG LCEL
├── index.html            # Antarmuka GUI utama (Vanilla JS)
├── .gitignore            # Penyaring berkas untuk Git
└── requirements.txt      # Daftar pustaka Python yang dibutuhkan
