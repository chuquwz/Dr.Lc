"""
Embedding Script cho Dr.Lc RAG Chatbot (Bản Local)
==================================================
Script này thực hiện bước 3 trong pipeline RAG:
1. Đọc tất cả chunks từ chunks.json.
2. Dùng thư viện sentence-transformers để tải và chạy mô hình embedding local
   (paraphrase-multilingual-MiniLM-L12-v2 - 384 chiều, hỗ trợ đa ngôn ngữ/tiếng Việt).
3. Lưu vectors + metadata vào ChromaDB.

TẠI SAO DÙNG LOCAL EMBEDDING?
- Hoàn toàn miễn phí, không cần mạng Internet.
- KHÔNG bị giới hạn rate limit (lỗi 429) của Google API.
- Tốc độ xử lý cực nhanh (chỉ mất ~1-2 phút cho toàn bộ 11,000 chunks).
- Tiết kiệm chi phí API, giúp ghi điểm tốt trong CV vì chứng tỏ hiểu cách tối ưu RAG.

Usage:
    python src/embedding/embed_chunks.py
"""

import sys
import io
import os
import json
import time

# Fix encoding cho Windows console (tiếng Việt)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

# ============================================================
# CẤU HÌNH
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHUNKS_FILE = os.path.join(PROJECT_ROOT, "data-processed", "chunks.json")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector-store")

# Tên collection trong ChromaDB
COLLECTION_NAME = "dr_lc_products"

# Mô hình local đa ngôn ngữ tốt nhất & gọn nhẹ của Hugging Face
# - paraphrase-multilingual-MiniLM-L12-v2: ~420MB
# - Số chiều vector đầu ra: 384 chiều
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# BATCH_SIZE: Số lượng chunks xử lý đồng thời.
# Vì chạy local trên máy của bạn nên chúng ta có thể đặt batch size lớn hơn (ví dụ 256)
# để SentenceTransformers xử lý song song trên CPU/GPU rất nhanh.
BATCH_SIZE = 256


# ============================================================
# KHỞI TẠO CLIENTS
# ============================================================

def init_embedding_model():
    """
    Khởi tạo mô hình embedding local từ thư viện SentenceTransformer.
    Trong lần chạy đầu tiên, mô hình ~420MB sẽ được tải về và lưu vào cache local.
    Các lần chạy tiếp theo sẽ load ngay lập tức từ ổ cứng.
    """
    print(f"🔄 Đang tải mô hình local: {EMBEDDING_MODEL}...")
    start_time = time.time()
    
    # SentenceTransformer tự động phát hiện GPU (CUDA) nếu có, ngược lại chạy trên CPU
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"✅ Tải mô hình thành công! Mất {time.time() - start_time:.1f} giây.")
    return model


def init_chroma_collection():
    """
    Khởi tạo ChromaDB lưu dữ liệu persistent (trên ổ cứng).
    Mỗi lần chạy lại sẽ xóa collection cũ đi để ghi đè sạch sẽ từ đầu.
    """
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"🗑️  Đã xóa collection cũ '{COLLECTION_NAME}'")
    except Exception:
        pass  # Collection chưa có
    
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # Đo khoảng cách bằng Cosine Similarity
    )
    
    print(f"✅ ChromaDB collection '{COLLECTION_NAME}' đã sẵn sàng.")
    return collection


# ============================================================
# HÀM CHÍNH
# ============================================================

def main():
    # ── Bước 1: Đọc chunks ───────────────────────────────────
    print(f"📂 Đọc file chunks: {CHUNKS_FILE}")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    print(f"   Tổng số chunks cần embed: {len(all_chunks)}")
    
    # ── Bước 2: Khởi tạo mô hình và Vector DB ──────────────────
    model = init_embedding_model()
    collection = init_chroma_collection()
    
    # ── Bước 3: Embedding & nạp vào Vector DB ──────────────────
    total_chunks = len(all_chunks)
    total_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n🚀 Bắt đầu tạo embeddings cho {total_chunks} chunks ({total_batches} batches, batch size {BATCH_SIZE})...")
    
    start_time = time.time()
    
    for batch_idx in range(total_batches):
        # Lấy slice của batch hiện tại
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, total_chunks)
        batch_chunks = all_chunks[batch_start:batch_end]
        
        # Chuẩn bị dữ liệu nạp vào DB
        ids = [chunk["chunk_id"] for chunk in batch_chunks]
        documents = [chunk["content"] for chunk in batch_chunks]
        metadatas = [
            {
                "product_sku": chunk["product_sku"],
                "product_name": chunk["product_name"],
                "product_url": chunk["product_url"],
                "section": chunk["section"],
            }
            for chunk in batch_chunks
        ]
        
        # Thực hiện embedding local
        # model.encode nhận list of string và trả về numpy array các vectors
        # show_progress_bar=False để tránh log rác terminal
        embeddings_ndarray = model.encode(documents, show_progress_bar=False)
        
        # Convert numpy array sang list các list floats để ChromaDB nhận diện
        embeddings = embeddings_ndarray.tolist()
        
        # Lưu vào ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        # In tiến trình
        elapsed = time.time() - start_time
        chunks_done = batch_end
        speed = chunks_done / elapsed if elapsed > 0 else 0
        eta = (total_chunks - chunks_done) / speed if speed > 0 else 0
        
        print(f"   ✅ Batch {batch_idx + 1}/{total_batches} "
              f"({chunks_done}/{total_chunks} chunks) "
              f"[{elapsed:.0f}s đã qua, ETA: ~{eta:.0f}s, Tốc độ: {speed:.0f} chunks/giây]")
        
    # ── Bước 4: Thống kê ─────────────────────────────────────
    total_time = time.time() - start_time
    db_count = collection.count()
    
    print(f"\n{'=' * 60}")
    print(f"🎉 HOÀN THÀNH EMBEDDING LOCAL!")
    print(f"{'=' * 60}")
    print(f"   Tổng chunks đã lưu:  {db_count}")
    print(f"   Mô hình embedding:   {EMBEDDING_MODEL}")
    print(f"   Số chiều vector:     384")
    print(f"   Tổng thời gian:      {total_time:.1f} giây")
    print(f"   Tốc độ trung bình:   {db_count / total_time:.1f} chunks/giây")
    print(f"   Thư mục Vector DB:   {VECTOR_STORE_DIR}")
    
    # ── Test nhanh: thử query local ──────────────────────────
    print(f"\n{'=' * 60}")
    print(f"🧪 CHẠY THỬ TRUY XUẤT (RETRIEVAL)...")
    print(f"{'=' * 60}")
    
    test_query = "sốt xuất huyết dùng thuốc gì tránh"
    print(f'   Câu hỏi test: "{test_query}"')
    
    # Embed câu hỏi test bằng model local
    query_embedding = model.encode([test_query])[0].tolist()
    
    # Tìm 3 chunks gần nhất trong ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )
    
    print(f"\n   Top 3 chunks liên quan nhất tìm thấy:")
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        distance = results["distances"][0][i]
        product_name = results["metadatas"][0][i]["product_name"]
        section = results["metadatas"][0][i]["section"]
        text_preview = results["documents"][0][i].replace('\n', ' ')[:100]
        
        # Cosine similarity = 1 - distance
        similarity = 1 - distance
        
        print(f"   {i+1}. [{similarity:.3f}] {product_name[:50]}... ({section})")
        print(f"      Nội dung: {text_preview}...")


if __name__ == "__main__":
    main()
