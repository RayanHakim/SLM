import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

def main():
    pdf_folder = "./data"
    
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        print("Folder './data' berhasil dibuat. Silakan tempatkan file PDF jurnal di sana.")
        return
        
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    if not pdf_files:
        print("Waduh, belum ada file PDF di folder ./data!")
        return

    print(f"Menemukan {len(pdf_files)} file PDF. Mulai membaca...")
    all_chunks = []
    
    for file in pdf_files:
        pdf_path = os.path.join(pdf_folder, file)
        print(f"Memproses file: {file}")
        
        try:
            loader = PDFPlumberLoader(pdf_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            for chunk in chunks:
                page_num = chunk.metadata.get('page', 0) + 1
                chunk.page_content = (
                    f"[INFO DOKUMEN] Judul File: {file} | Halaman: {page_num}\n"
                    f"[ISI TEKS]:\n{chunk.page_content}"
                )
            
            all_chunks.extend(chunks)
            print(f"Sukses ekstraksi {len(chunks)} chunks dari {file}.")
        except Exception as e:
            print(f"Gagal memproses {file}. Error: {str(e)}")

    print(f"\nTotal akumulasi data: {len(all_chunks)} chunks.")

    print("Inisialisasi model embedding 'nomic-embed-text' via Ollama...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # PERBAIKAN LOKAL: Menyimpan data langsung ke folder lokal tanpa server Docker
    print("Menyimpan representasi vektor ke Qdrant Lokal (./qdrant_db)...")
    QdrantVectorStore.from_documents(
        all_chunks,
        embeddings,
        path="./qdrant_db",
        collection_name="jurnal_koleksi"
    )
    
    print("\n🎉 Selesai! Database vektor Qdrant lokal berhasil dibuat.")

if __name__ == "__main__":
    main()