# RAG Chatbot using Phi-3 Mini & FastAPI

Proyek ini adalah sistem *Retrieval-Augmented Generation* (RAG) yang berjalan 100% secara lokal dan luring (offline). Sistem ini dirancang untuk melakukan analisis dokumen teks, jurnal ilmiah, hingga artikel web secara aman tanpa membocorkan data ke pihak ketiga. Dilengkapi dengan fitur **Dual Mode**, pengguna dapat secara dinamis memilih antara basis pengetahuan dokumen lokal yang ketat atau memanfaatkan memori pengetahuan umum bawaan dari model bahasa secara luwes.

## 🚀 Fitur Utama

- **100% Local & Offline Execution**: Memanfaatkan fungsionalitas model *Small Language Model* (SLM) Phi-3 Mini (3.8B) via Ollama yang berjalan langsung di atas infrastruktur GPU lokal (VRAM ramah lingkungan).
- **Dynamic Dual-Mode Knowledge Base**:
  - **Dokumen Lokal (RAG)**: AI dipaksa secara ketat hanya menjawab berdasarkan konteks dokumen atau artikel hasil *scraping* di dalam database vektor. Jika informasi tidak ditemukan, AI akan menolak secara jujur guna menghindari halusinasi.
  - **Pengetahuan Umum**: Mem-bypass database vektor dan membebaskan Phi-3 Mini menjawab menggunakan memori internal bawaannya untuk penalaran umum atau pembuatan kode pemrograman.
- **Automated Web & Document Ingestion**:
  - `ingest.py`: Mengotomatisasi pembacaan, pemotongan teks (*chunking*), dan ekstraksi metadata halaman dari dokumen PDF lokal secara cepat.
  - `auto_scrap_ingest.py`: Memanfaatkan mesin asinkronous **Playwright** dan **BeautifulSoup** untuk menyedot konten artikel web publik secara bersih langsung ke database vektor.
- **Traceability & Citation**: Menyediakan transparansi analisis dengan melampirkan informasi dokumen sumber (*source*) beserta indeks bagian/halaman tempat data semantik diekstraksi.
- **FastAPI Lifespan Management**: Manajemen koneksi database Qdrant yang aman di lingkup *global scope* menggunakan FastAPI Lifespan Context Manager, mencegah terjadinya *permission error lock* akibat proses *reloader* Uvicorn di Windows.

## 🛠️ Tech Stack

- **Backend Framework**: FastAPI & Uvicorn (Python 3.12)
- **Vector Database**: Qdrant (Local Path Storage)
- **LLM & Embeddings**: Phi-3 Mini (3.8B) & Nomic Embed Text via **LangChain Ollama** (Modul Terbaru)
- **Web Scraper Engine**: Playwright (Async API) & BeautifulSoup4
- **PDF Parser**: PDFPlumber
- **Frontend**: HTML5, CSS3 (Slate & Teal Theme), Vanilla JavaScript (Fetch API)

---

🛠️ Cara Menjalankan Proyek Setelah Di-Clone
Ikuti panduan langkah demi langkah berikut untuk menjalankan platform RAG lokal ini di komputer Anda setelah melakukan git clone:

1. Prasyarat Sistem (Prerequisites)
Pastikan komputer Anda sudah terpasang:

Python 3.10 atau versi di atasnya

Ollama (Sudah mengunduh model melalui perintah: ollama pull phi3 dan ollama pull nomic-embed-text)

2. Persiapan Repositori & Virtual Environment
Buka terminal (PowerShell/CMD), masuk ke dalam direktori proyek, lalu buat virtual environment terisolasi:

Bash
# Masuk ke folder proyek
cd SLM

# Membuat virtual environment bernama 'env'
python -m venv env

# Aktivasi virtual environment (Windows PowerShell)
.\env\Scripts\Activate.ps1
3. Instalasi Pustaka & Browser Binaries
Instal seluruh dependensi Python yang dibutuhkan, serta pasang mesin headless browser untuk kebutuhan scraping data web:

Bash
# Upgrade pip dan instal library utama
pip install -r requirements.txt

# Wajib: Instal binary browser Chromium milik Playwright
playwright install chromium
4. Proses Ingest Data (Mengisi Otak Database)
Anda bisa memasukkan pengetahuan ke database lewat dua jalur lokal:

Jalur A (File PDF Lokal): Masukkan berkas PDF Anda ke dalam folder data/, lalu jalankan:

Bash
python ingest.py
Jalur B (Artikel Internet): Ubah variabel target_url di dalam file auto_scrap_ingest.py ke alamat web yang diinginkan, lalu jalankan:

Bash
python auto_scrap_ingest.py
5. Menyalakan Server Backend & Akses GUI
Setelah database vektor terisi di folder qdrant_db, nyalakan server FastAPI menggunakan Uvicorn:

Bash
uvicorn api:app --reload
Setelah server sukses berjalan (menampilkan log Sistem RAG dan database Qdrant siap digunakan!), buka browser kesayangan Anda dan akses alamat:

Plaintext
[http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## 📂 Struktur Proyek

```text
SLM/
├── data/                 # Tempat menyimpan berkas PDF lokal (misal: Jurnal Skripsi)
├── qdrant_db/            # Database vektor biner lokal (Diabaikan oleh Git via .gitignore)
├── ingest.py             # Skrip ekstraksi dan vektorisasi dokumen PDF lokal ke Qdrant
├── auto_scrap_ingest.py  # Skrip scraping asinkronous artikel internet ke Qdrant
├── api.py                # Server backend FastAPI, arsitektur Lifespan, & Pipeline Dual Mode
├── index.html            # Antarmuka dasbor GUI utama (Modern Dark Mode)
├── .gitignore            # Berkas penyaring agar file biner database tidak terunggah
└── requirements.txt      # Daftar dependensi pustaka Python pihak ketiga
