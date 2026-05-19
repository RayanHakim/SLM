import os
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

async def scrape_website(url):
    print(f"Membuka website: {url} ...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            html_content = await page.content()
            title = await page.title()
            await browser.close()
            return title, html_content
        except Exception as e:
            print(f"Gagal mengambil data dari URL. Error: {str(e)}")
            await browser.close()
            return None, None

def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
        
    clean_text = soup.get_text(separator="\n")
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    return "\n".join(lines)

async def main():
    target_url = "https://id.wikipedia.org/wiki/Toyota"
    
    # 1. Proses Scraping
    title, raw_html = await scrape_website(target_url)
    if not raw_html:
        return
        
    text_content = clean_html(raw_html)
    print(f"Sukses mengekstrak teks dari artikel: '{title}'")
    
    # 2. Proses Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text_content)
    
    # IMPROVEMENT: page_content murni teks bersih agar embedding lebih akurat, metadata dipisah
    documents = []
    for i, chunk_text in enumerate(chunks):
        doc = Document(
            page_content=chunk_text,
            metadata={
                "source": target_url,
                "title": title,
                "page": i + 1  # Kita petakan chunk_index ke 'page' agar kompatibel dengan api.py Anda
            }
        )
        documents.append(doc)
        
    print(f"Teks berhasil dipotong menjadi {len(documents)} chunks.")
    
    # 3. Inisialisasi Model Embedding (versi langchain-ollama)
    print("Inisialisasi model embedding 'nomic-embed-text' via Ollama...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    print("Menyimpan hasil scrap internet ke Qdrant Lokal (./qdrant_db)...")
    
    # SOLUSI UTAMA: Menggunakan fungsi bawaan dari_documents dengan parameter path langsung.
    # Ini otomatis membuat database dan koleksi baru jika belum terdeteksi di folder proyek.
    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        path="./qdrant_db",
        collection_name="jurnal_koleksi"
    )
    
    print(f"\n Selesai! Data dari internet berhasil disuntikkan ke database Qdrant lokal Anda.")

if __name__ == "__main__":
    asyncio.run(main())