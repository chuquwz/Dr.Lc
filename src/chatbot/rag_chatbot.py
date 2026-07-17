"""
Dr.Lc RAG Chatbot — Phiên bản Chat-RAG hoàn chỉnh (Hybrid Retrieval + RRF + Lịch sử Hội thoại)
========================================================================================
Script này thực hiện sự kết hợp nâng cao nhất của pipeline RAG chạy Local (Không Reranker):
1. Nhận câu hỏi từ người dùng qua terminal.
2. Dùng LLM local (Ollama) viết lại câu hỏi thô thành Standalone Query (câu hỏi độc lập) dựa vào Lịch sử Hội thoại.
3. Chạy song song 2 luồng truy xuất bằng Standalone Query:
   - Dense Search (Vector): Dùng SentenceTransformer + ChromaDB để lấy Top-20 chunks theo ngữ nghĩa.
   - Sparse Search (Từ khóa): Dùng thuật toán BM25Okapi local để lấy Top-20 chunks khớp từ khóa chính xác.
4. Gộp và xếp hạng lại kết quả bằng thuật toán RRF (Reciprocal Rank Fusion) với hằng số K=60.
5. Lấy Top-4 chunks có điểm RRF cao nhất làm ngữ cảnh đáng tin cậy.
6. Xây dựng prompt y tế an toàn và gọi API `/api/chat` của Ollama ở chế độ Streaming để trả lời thời gian thực.
7. Hiển thị chữ chạy, in nguồn tham khảo y khoa và cập nhật lịch sử chat.

Usage:
    python src/chatbot/rag_chatbot.py
"""

import sys
import io
import os
import json
import re
import time
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# Fix encoding cho Windows console (tiếng Việt không lỗi)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# ============================================================
# CẤU HÌNH & LOAD MÔI TRƯỜNG
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CHUNKS_FILE = os.path.join(PROJECT_ROOT, "data-processed", "chunks.json")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector-store")
COLLECTION_NAME = "dr_lc_products"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Ollama API URL cho Chat và Generate
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
LOCAL_LLM_MODEL = "qwen3:4b"

# Số lượng lượt hội thoại tối đa được lưu trong bộ nhớ (tránh quá tải ngữ cảnh)
MAX_HISTORY_TURNS = 5


# ============================================================
# HÀM BỔ TRỢ: TOKENIZER CHO TIẾNG VIỆT (DÙNG CHO BM25)
# ============================================================

def clean_and_tokenize(text: str) -> list[str]:
    """
    Tách từ (tokenizer) đơn giản cho tiếng Việt để dùng cho BM25.
    """
    cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = [token for token in cleaned_text.split() if token.strip()]
    return tokens


# ============================================================
# KHỞI TẠO MÔ HÌNH & DATABASE
# ============================================================

def init_components():
    """
    Khởi tạo đồng thời:
    1. Kiểm tra kết nối tới Ollama local
    2. Load mô hình embedding local (SentenceTransformer)
    3. Kết nối tới ChromaDB local (Dense Search)
    4. Load file chunks.json và build chỉ mục BM25 (Sparse Search)
    """
    # 1. Kiểm tra kết nối Ollama
    try:
        response = requests.get("http://localhost:11434/", timeout=2)
        if response.status_code == 200:
            print("✅ Đã kết nối thành công tới Ollama (localhost:11434)")
    except requests.exceptions.ConnectionError:
        print("❌ LỖI: Không thể kết nối tới Ollama!")
        print("   Vui lòng mở ứng dụng Ollama trên máy của bạn trước.")
        print("   Hoặc chạy lệnh 'ollama serve' trong cmd.")
        sys.exit(1)
        
    # 2. Khởi tạo Embedding Model Local
    print("🔄 Đang load mô hình embedding local (SentenceTransformer)...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    
    # 3. Khởi tạo ChromaDB
    print("🔄 Đang kết nối tới Vector Database (ChromaDB)...")
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ LỖI: Không tìm thấy collection '{COLLECTION_NAME}' trong ChromaDB!")
        print("   Vui lòng chạy script 'src/embedding/embed_chunks.py' trước để nạp dữ liệu.")
        sys.exit(1)
        
    # 4. Load chunks và xây dựng chỉ mục BM25
    print(f"🔄 Đang đọc file chunks và xây dựng chỉ mục BM25...")
    start_time = time.time()
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
        
    # Tokenize toàn bộ corpus chunks
    tokenized_corpus = [clean_and_tokenize(chunk["content"]) for chunk in all_chunks]
    
    # Khởi tạo BM25Okapi (thuật toán BM25 chuẩn)
    bm25_model = BM25Okapi(tokenized_corpus)
    
    print(f"✅ Đã dựng xong chỉ mục BM25 cho {len(all_chunks)} chunks trong {time.time() - start_time:.2f} giây.")
    return embed_model, collection, all_chunks, bm25_model


# ============================================================
# BƯỚC 1: CONVERSE & CONDENSE QUERY (TÓM TẮT & VIẾT LẠI CÂU HỎI)
# ============================================================

def condense_query(raw_query: str, chat_history: list[dict]) -> str:
    """
    Dựa vào lịch sử chat và câu hỏi thô mới, yêu cầu LLM viết lại thành 
    một câu tìm kiếm độc lập (Standalone Query) chứa đầy đủ ngữ cảnh cũ.
    """
    if not chat_history:
        return raw_query
        
    history_str = ""
    for msg in chat_history[-MAX_HISTORY_TURNS*2:]:
        role_label = "Người dùng" if msg["role"] == "user" else "Trợ lý Dược sĩ"
        history_str += f"{role_label}: {msg['content']}\n"
        
    system_prompt = (
        "Nhiệm vụ của bạn là đọc LỊCH SỬ HỘI THOẠI và CÂU HỎI MỚI dưới đây, sau đó viết lại câu hỏi "
        "thành một câu hỏi tìm kiếm ĐỘC LẬP, ĐẦY ĐỦ Ý NGHĨA (không dùng từ thay thế mơ hồ như 'nó', 'chúng', 'cho tôi', 'như thế nào').\n"
        "Chỉ trả về câu hỏi đã được viết lại bằng tiếng Việt, không giải thích gì thêm."
    )
    
    prompt = (
        f"LỊCH SỬ HỘI THOẠI:\n"
        f"{history_str}\n"
        f"CÂU HỎI MỚI: {raw_query}\n\n"
        f"CÂU HỎI ĐỘC LẬP SAU KHI VIẾT LẠI:"
    )
    
    payload = {
        "model": LOCAL_LLM_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    
    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=20)
        if response.status_code == 200:
            standalone_query = response.json().get("response", "").strip()
            standalone_query = standalone_query.strip('"').strip("'")
            return standalone_query
    except Exception as e:
        pass
        
    return raw_query


# ============================================================
# BƯỚC 2: HYBRID RETRIEVAL & RRF (TRUY XUẤT HỖN HỢP)
# ============================================================

def hybrid_retrieve(query: str, embed_model, collection, all_chunks, bm25_model, k: int = 4) -> list[dict]:
    """
    Thực hiện Hybrid Retrieval kết hợp Dense (Vector) + Sparse (BM25) sử dụng RRF.
    """
    # ── 1. DENSE SEARCH (ChromaDB) ───────────────────────────
    query_vector = embed_model.encode([query], show_progress_bar=False)[0].tolist()
    dense_results = collection.query(
        query_embeddings=[query_vector],
        n_results=20
    )
    
    dense_hits = []
    if dense_results["ids"] and dense_results["ids"][0]:
        ids = dense_results["ids"][0]
        documents = dense_results["documents"][0]
        metadatas = dense_results["metadatas"][0]
        distances = dense_results["distances"][0]
        
        for i in range(len(ids)):
            dense_hits.append({
                "chunk_id": ids[i],
                "content": documents[i],
                "metadata": metadatas[i],
                "similarity": 1 - distances[i]
            })

    # ── 2. SPARSE SEARCH (BM25) ──────────────────────────────
    query_tokens = clean_and_tokenize(query)
    bm25_scores = bm25_model.get_scores(query_tokens)
    
    sparse_indices = [idx for idx, score in enumerate(bm25_scores) if score > 0]
    sparse_indices = sorted(sparse_indices, key=lambda idx: bm25_scores[idx], reverse=True)[:20]
    
    sparse_hits = []
    for idx in sparse_indices:
        chunk = all_chunks[idx]
        sparse_hits.append({
            "chunk_id": chunk["chunk_id"],
            "content": chunk["content"],
            "metadata": {
                "product_sku": chunk["product_sku"],
                "product_name": chunk["product_name"],
                "product_url": chunk["product_url"],
                "section": chunk["section"]
            },
            "bm25_score": bm25_scores[idx]
        })

    # ── 3. RECIPROCAL RANK FUSION (RRF) ──────────────────────
    RRF_K = 60
    rrf_scores = {}
    chunks_map = {}
    
    for rank, hit in enumerate(dense_hits):
        cid = hit["chunk_id"]
        chunks_map[cid] = hit
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_K + (rank + 1)))
        
    for rank, hit in enumerate(sparse_hits):
        cid = hit["chunk_id"]
        chunks_map[cid] = hit
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_K + (rank + 1)))

    # ── 4. SẮP XẾP & LỌC TOP-K KẾT QUẢ ────────────────────────
    sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    
    final_chunks = []
    for cid in sorted_cids[:k]:
        chunk_info = chunks_map[cid]
        final_chunks.append({
            "id": cid,
            "content": chunk_info["content"],
            "metadata": chunk_info["metadata"],
            "rrf_score": rrf_scores[cid],
            "source_type": ("Hybrid" if (cid in [d["chunk_id"] for d in dense_hits] and cid in [s["chunk_id"] for s in sparse_hits]) 
                            else "Dense" if cid in [d["chunk_id"] for d in dense_hits] 
                            else "Sparse")
        })
        
    return final_chunks


# ============================================================
# BƯỚC 3: GENERATION (GỌI /API/CHAT OLLAMA DẠNG STREAMING)
# ============================================================

def generate_chat_stream(query: str, retrieved_chunks: list[dict], chat_history: list[dict]):
    """
    Sử dụng endpoint `/api/chat` để duy trì lịch sử hội thoại.
    Tích hợp prompt y tế và ngữ cảnh y khoa vào tin nhắn hiện tại.
    """
    context_items = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        p_name = meta.get("product_name", "Thuốc")
        context_items.append(
            f"--- TÀI LIỆU {i} (Sản phẩm: {p_name} | Nguồn tìm thấy: {chunk['source_type']}) ---\n"
            f"{chunk['content']}\n"
        )
    context_str = "\n".join(context_items)
    
    system_instruction = (
        "Bạn là Dr.Lc (Doctor Long Châu) - một trợ lý dược sĩ ảo thông minh chuyên tư vấn y tế "
        "và hỗ trợ mua thuốc dựa trên dữ liệu sản phẩm của nhà thuốc Long Châu.\n\n"
        "Nhiệm vụ của bạn:\n"
        "1. Trả lời câu hỏi của người dùng một cách chính xác, chuyên nghiệp, dễ hiểu bằng tiếng Việt.\n"
        "2. PHÂN LOẠI CÂU HỎI & XỬ LÝ NGUỒN TRI THỨC:\n"
        "   - THỂ A (Hỏi về Thuốc - Ví dụ: liều dùng, tác dụng phụ, giá cả, cách uống của một thuốc cụ thể): Bạn BẮT BUỘC chỉ sử dụng thông tin "
        "được cung cấp trong phần 'NGỮ CẢNH TRUY XUẤT' dưới đây. Tuyệt đối không tự ý bịa thông tin thuốc nằm ngoài ngữ cảnh.\n"
        "   - THỂ B (Hỏi về Triệu chứng, Bệnh học hoặc Lời khuyên sức khỏe - Ví dụ: đau bụng là bệnh gì, tôi nên làm gì): "
        "Vì phần 'NGỮ CẢNH TRUY XUẤT' chỉ chứa thông tin thuốc, bạn ĐƯỢC PHÉP sử dụng tri thức y khoa rộng lớn của mình để giải thích "
        "các nguyên nhân phổ biến (như thiếu ngủ, đầy hơi, huyết áp...) và đưa ra lời khuyên lối sống. TUYỆT ĐỐI không khuyên dùng "
        "các thuốc nằm ngoài Ngữ cảnh truy xuất, chỉ được tư vấn hướng đi khám.\n"
        "   - THỂ C (Hỏi ngoài phạm vi - Không liên quan đến y tế, sức khỏe, bệnh tật, lối sống lành mạnh hay thông tin thuốc. Ví dụ: thời tiết, công nghệ, toán học, viết code, dịch thuật, thơ ca...): "
        "Bạn BẮT BUỘC phải từ chối trả lời một cách lịch sự và nói chính xác câu sau: 'Tôi xin lỗi, câu hỏi này nằm ngoài phạm vi hỗ trợ của trợ lý y tế Dr.Lc. Tôi chỉ có thể tư vấn các vấn đề liên quan đến y tế, sức khỏe, bệnh tật và thông tin thuốc.'\n"
        "3. Nếu câu hỏi thuộc THỂ A nhưng NGỮ CẢNH không chứa thông tin, hãy trả lời rõ là: "
        "'Tôi xin lỗi, thông tin về sản phẩm này hiện không có trong dữ liệu của nhà thuốc Long Châu. "
        "Để đảm bảo an toàn, bạn nên tham khảo ý kiến của bác sĩ hoặc dược sĩ chuyên môn.'\n"
        "4. CẢNH BÁO AN TOÀN: Khi tư vấn y tế, luôn khuyên người dùng đi khám bác sĩ để được chẩn đoán chính xác."
    )
    
    messages = []
    messages.append({"role": "system", "content": system_instruction})
    messages.extend(chat_history[-MAX_HISTORY_TURNS*2:])
    
    current_content = (
        f"NGỮ CẢNH TRUY XUẤT:\n"
        f"{context_str}\n\n"
        f"CÂU HỎI HIỆN TẠI:\n"
        f"{query}"
    )
    messages.append({"role": "user", "content": current_content})
    
    payload = {
        "model": LOCAL_LLM_MODEL,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.15
        }
    }
    
    try:
        response = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=90, stream=True)
        
        if response.status_code != 200:
            yield (
                f"⚠️ Lỗi Ollama (HTTP {response.status_code}): {response.text}\n\n"
                f"💡 Gợi ý khắc phục: Đổi model lại thành một model nhẹ hơn hoặc khởi động lại Ollama."
            )
            return
            
        for line in response.iter_lines():
            if line:
                res_json = json.loads(line.decode('utf-8'))
                
                if "error" in res_json:
                    yield f"⚠️ Lỗi từ Ollama API: {res_json['error']}"
                    return
                
                message_chunk = res_json.get("message", {})
                token = message_chunk.get("content", "")
                yield token
                
                if res_json.get("done", False):
                    break
                    
    except requests.exceptions.Timeout:
        yield "⚠️ Lỗi: Thời gian phản hồi từ Ollama quá hạn (Timeout). Bạn hãy thử lại."
    except Exception as e:
        yield f"⚠️ Lỗi kết nối tới Ollama: {e}"


# ============================================================
# CHƯƠNG TRÌNH CHÍNH (GIAO DIỆN CONSOLE)
# ============================================================

def main():
    print("=" * 70)
    print("🏥 DR.LC CHATBOT — PHIÊN BẢN HYBRID RETRIEVAL (DENSE + SPARSE) + RRF")
    print("=" * 70)
    
    # Khởi tạo các thành phần
    embed_model, collection, all_chunks, bm25_model = init_components()
    
    print(f"\nSystem: Đang sử dụng mô hình local: '{LOCAL_LLM_MODEL}' qua Ollama.")
    print("System: Công nghệ tìm kiếm: Hybrid Search (ChromaDB Vector + BM25 Local) + Xếp hạng RRF.")
    print("System: Gõ 'exit' hoặc 'quit' để thoát chương trình.")
    print("-" * 70)
    
    chat_history = []
    symptom_keywords = ["là dấu hiệu", "là bệnh gì", "bị bệnh gì", "nguyên nhân", "dấu hiệu của", "triệu chứng của"]
    
    while True:
        try:
            user_query = input("\n👤 Bạn: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit"]:
                print("👋 Tạm biệt bạn! Chúc bạn nhiều sức khỏe!")
                break
                
            print("🔍 Dr.Lc đang phân tích ngữ cảnh câu hỏi...")
            standalone_query = condense_query(user_query, chat_history)
            
            if standalone_query != user_query:
                print(f"   [Debug: Câu hỏi đã được làm rõ thành -> '{standalone_query}']")
            
            print("🤖 Dr.Lc đang truy xuất thông tin (Hybrid + RRF)...")
            chunks = hybrid_retrieve(standalone_query, embed_model, collection, all_chunks, bm25_model, k=4)
            
            if not chunks:
                print("🤖 Dr.Lc: Tôi xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi trong dữ liệu của nhà thuốc Long Châu.")
                continue
            
            print("   [Debug: Top Chunks tìm thấy bằng RRF]")
            for i, c in enumerate(chunks):
                print(f"     {i+1}. Score: {c['rrf_score']:.4f} | Nguồn: {c['source_type']:6s} | {c['metadata']['product_name'][:40]}... ({c['metadata']['section']})")
            
            print("🤖 Dr.Lc đang suy nghĩ trả lời (chạy local)...")
            print("\n🤖 Dr.Lc: ", end="", flush=True)
            
            full_response = []
            for token in generate_chat_stream(user_query, chunks, chat_history):
                print(token, end="", flush=True)
                full_response.append(token)
                
            full_text = "".join(full_response)
            
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": full_text})
            
            is_symptom_query = any(kw in standalone_query.lower() for kw in symptom_keywords)
            is_out_of_scope = "nằm ngoài phạm vi hỗ trợ" in full_text
            if not is_symptom_query and not is_out_of_scope and "Tôi xin lỗi" not in full_text and "Lỗi" not in full_text:
                sources = {}
                for chunk in chunks:
                    meta = chunk["metadata"]
                    p_name = meta.get("product_name", "Thuốc")
                    p_url = meta.get("product_url", "")
                    if p_name and p_url:
                        sources[p_name] = p_url
                
                if sources:
                    print("\n\n🔗 **Thông tin sản phẩm tham khảo:**", end="")
                    for name, url in sources.items():
                        short_name = name.split("(")[0].strip() if "(" in name else name
                        print(f"\n- [{short_name}]({url})", end="")
            
            print()
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt bạn!")
            break
        except Exception as e:
            print(f"\n❌ LỖI HỆ THỐNG: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()
