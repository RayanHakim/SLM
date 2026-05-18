import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def main():
    pdf_folder = "./data"
    
    # Buat folder data kalau belum ada
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        print("Folder './data' berhasil dibuat. Taruh PDF laporan keuangan di sana dulu, Le!")
        return
        
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    if not pdf_files:
        print("Waduh, belum ada file PDF di folder ./data!")
        return

    print(f"Menemukan {len(pdf_files)} file PDF. Mulai membaca...")
    all_chunks = []
    
    # Loop untuk baca dan potong-potong teks PDF (Chunking)
    for file in pdf_files:
        pdf_path = os.path.join(pdf_folder, file)
        print(f"Proses file: {file}")
        
        try:
            loader = PDFPlumberLoader(pdf_path)
            documents = loader.load()
            
            # Potong per 1000 karakter dengan overlap 200 karakter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            print(f"Sukses ekstrak {len(chunks)} chunks dari {file}.")
        except Exception as e:
            print(f"Gagal memproses {file}. Error: {str(e)}")

    print(f"\nTotal akumulasi data: {len(all_chunks)} chunks.")

    # Hubungkan ke model embedding local via Ollama
    print("Inisialisasi model embedding Phi-3 via Ollama...")
    embeddings = OllamaEmbeddings(model="phi3")
    
    # Generate vektor dan simpan ke database ChromaDB lokal
    print("Menyimpan representasi vektor ke ./chroma_db...")
    db = Chroma.from_documents(all_chunks, embeddings, persist_directory="./chroma_db")
    
    print("\n🎉 Selesai! Database vektor berhasil dibuat.")

if __name__ == "__main__":
    main()