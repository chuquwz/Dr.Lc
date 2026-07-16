"""
Dr.Lc RAG Chatbot — Phiên bản Chatbot RAG hoàn chỉnh (Có Lịch sử Hội thoại)
========================================================================
Script này thực hiện Bước 4.3 trong lộ trình:
1. Nhận câu hỏi từ người dùng qua terminal.
2. Nếu có lịch sử chat, gọi LLM local (Ollama) viết lại câu hỏi thô thành
   câu hỏi độc lập đầy đủ nghĩa (Standalone Query) dựa vào lịch sử hội thoại.
3. Dùng Standalone Query đó để truy xuất Top-4 chunks liên quan từ ChromaDB.
4. Xây dựng danh sách tin nhắn (messages) chứa: System Instruction, Lịch sử chat, Ngữ cảnh y tế và Câu hỏi mới.
5. Gọi API `/api/chat` của Ollama ở chế độ STREAMING để sinh câu trả lời ghi nhớ ngữ cảnh.
6. Hiển thị chữ chạy thời gian thực và cập nhật lịch sử chat.

Usage:
    python src/chatbot/rag_chatbot.py
"""

import sys
import io
import os
import json
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

# Fix encoding cho Windows console (tiếng Việt không lỗi)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# ============================================================
# CẤU HÌNH & LOAD MÔI TRƯỜNG
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector-store")
COLLECTION_NAME = "dr_lc_products"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Ollama API URL cho Chat
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
LOCAL_LLM_MODEL = "qwen3:4b"

# Số lượng lượt hội thoại tối đa được lưu trong bộ nhớ (tránh quá tải ngữ cảnh)
MAX_HISTORY_TURNS = 5


# ============================================================
# KHỞI TẠO MÔ HÌNH & DATABASE
# ============================================================

def init_components():
    """
    Khởi tạo kết nối tới Ollama, load SentenceTransformer và kết nối tới ChromaDB.
    """
    try:
        response = requests.get("http://localhost:11434/", timeout=2)
        if response.status_code == 200:
            print("✅ Đã kết nối thành công tới Ollama (localhost:11434)")
    except requests.exceptions.ConnectionError:
        print("❌ LỖI: Không thể kết nối tới Ollama!")
        print("   Vui lòng mở ứng dụng Ollama trên máy của bạn trước.")
        sys.exit(1)
        
    print("🔄 Đang load mô hình embedding local (SentenceTransformer)...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("🔄 Đang kết nối tới Vector Database (ChromaDB)...")
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ LỖI: Không tìm thấy collection '{COLLECTION_NAME}' trong ChromaDB!")
        sys.exit(1)
        
    print("✅ Các thành phần đã sẵn sàng!")
    return embed_model, collection


# ============================================================
# BƯỚC 1: CONVERSE & CONDENSE QUERY (TÓM TẮT & VIẾT LẠI CÂU HỎI)
# ============================================================

def condense_query(raw_query: str, chat_history: list[dict]) -> str:
    """
    Dựa vào lịch sử chat và câu hỏi thô mới, yêu cầu LLM viết lại thành 
    một câu tìm kiếm độc lập (Standalone Query) chứa đầy đủ ngữ cảnh cũ.
    
    Ví dụ:
      - Lịch sử: User: "đau đầu chóng mặt là dấu hiệu của gì" -> Bot trả lời.
      - User hỏi mới: "kê đơn cho tôi"
      - Standalone Query viết lại: "Thuốc kê đơn điều trị triệu chứng đau đầu chóng mặt"
    """
    # Nếu chưa có lịch sử hội thoại, không cần viết lại
    if not chat_history:
        return raw_query
        
    # Tạo chuỗi lịch sử chat rút gọn cho LLM đọc
    history_str = ""
    for msg in chat_history[-MAX_HISTORY_TURNS*2:]: # Lấy tối đa số lượt cấu hình
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
            "temperature": 0.0 # temperature = 0.0 để viết lại chính xác nhất
        }
    }
    
    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=20)
        if response.status_code == 200:
            standalone_query = response.json().get("response", "").strip()
            # Làm sạch nếu LLM lỡ bao bọc câu hỏi trong ngoặc kép
            standalone_query = standalone_query.strip('"').strip("'")
            return standalone_query
    except Exception as e:
        # Nếu lỗi viết lại, fallback dùng câu hỏi thô ban đầu để tránh crash
        pass
        
    return raw_query


# ============================================================
# BƯỚC 2: RETRIEVAL (TRUY XUẤT VECTOR VỚI STANDALONE QUERY)
# ============================================================

def retrieve_chunks(query: str, embed_model, collection, k: int = 4) -> list[dict]:
    """
    Truy xuất Top-K chunks từ ChromaDB bằng vector của Standalone Query.
    """
    query_vector = embed_model.encode([query], show_progress_bar=False)[0].tolist()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k
    )
    
    retrieved_data = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    for i in range(len(ids)):
        similarity = 1 - distances[i]
        retrieved_data.append({
            "id": ids[i],
            "content": documents[i],
            "metadata": metadatas[i],
            "similarity": similarity
        })
    return retrieved_data


# ============================================================
# BƯỚC 3: GENERATION (GỌI /API/CHAT OLLAMA DẠNG STREAMING)
# ============================================================

def generate_chat_stream(query: str, retrieved_chunks: list[dict], chat_history: list[dict]):
    """
    Sử dụng endpoint `/api/chat` để duy trì lịch sử hội thoại.
    Tích hợp prompt y tế và ngữ cảnh y khoa vào tin nhắn hiện tại.
    """
    # ── 1. Chuẩn bị khối ngữ cảnh từ các chunk đã tìm thấy ──────
    context_items = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        p_name = meta.get("product_name", "Thuốc")
        context_items.append(
            f"--- TÀI LIỆU {i} (Sản phẩm: {p_name}) ---\n"
            f"{chunk['content']}\n"
        )
    context_str = "\n".join(context_items)
    
    # ── 2. Xây dựng System Instruction ────────────────────────
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
        "3. Nếu câu hỏi thuộc THỂ A nhưng NGỮ CẢNH không chứa thông tin, hãy trả lời rõ là: "
        "'Tôi xin lỗi, thông tin về sản phẩm này hiện không có trong dữ liệu của nhà thuốc Long Châu. "
        "Để đảm bảo an toàn, bạn nên tham khảo ý kiến của bác sĩ hoặc dược sĩ chuyên môn.'\n"
        "4. CẢNH BÁO AN TOÀN: Khi tư vấn y tế, luôn khuyên người dùng đi khám bác sĩ để được chẩn đoán chính xác."
    )
    
    # ── 3. Xây dựng mảng Messages gửi đi ──────────────────────
    messages = []
    
    # a. Thêm System Message ở đầu
    messages.append({"role": "system", "content": system_instruction})
    
    # b. Thêm lịch sử chat từ bộ nhớ trượt (lấy tối đa MAX_HISTORY_TURNS lượt gần nhất)
    # Lịch sử chat được lưu dưới dạng [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
    messages.extend(chat_history[-MAX_HISTORY_TURNS*2:])
    
    # c. Thêm tin nhắn hiện tại kèm theo Ngữ cảnh y tế mới truy xuất được
    current_content = (
        f"NGỮ CẢNH TRUY XUẤT:\n"
        f"{context_str}\n\n"
        f"CÂU HỎI HIỆN TẠI:\n"
        f"{query}"
    )
    messages.append({"role": "user", "content": current_content})
    
    # ── 4. Gọi API /api/chat của Ollama ────────────────────────
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
                f"💡 Gợi ý khắc phục: Đổi model lại thành 'qwen2.5:3b' hoặc khởi động lại Ollama."
            )
            return
            
        for line in response.iter_lines():
            if line:
                res_json = json.loads(line.decode('utf-8'))
                
                if "error" in res_json:
                    yield f"⚠️ Lỗi từ Ollama API: {res_json['error']}"
                    return
                
                # Cấu trúc của /api/chat trả về token trong key 'message' -> 'content'
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
    print("=" * 60)
    print("🏥 DR.LC CHATBOT — PHIÊN BẢN CHAT-RAG HOÀN CHỈNH (OLLAMA)")
    print("=" * 60)
    
    embed_model, collection = init_components()
    
    print(f"\nSystem: Đang sử dụng mô hình local: '{LOCAL_LLM_MODEL}' qua Ollama.")
    print("System: Hệ thống đã bật bộ nhớ hội thoại. Chatbot sẽ nhớ được ngữ cảnh trò chuyện!")
    print("System: Gõ 'exit' hoặc 'quit' để thoát chương trình.")
    print("-" * 60)
    
    # Khởi tạo Lịch sử Hội thoại
    chat_history = []
    
    symptom_keywords = ["là dấu hiệu", "là bệnh gì", "bị bệnh gì", "nguyên nhân", "dấu hiệu của", "triệu chứng của"]
    
    while True:
        try:
            # Nhận query thô từ user
            user_query = input("\n👤 Bạn: ").strip()
            
            if not user_query:
                continue
                
            if user_query.lower() in ["exit", "quit"]:
                print("👋 Tạm biệt bạn! Chúc bạn nhiều sức khỏe!")
                break
                
            # 1. Condense Query: Viết lại câu hỏi thô thành Standalone Query dựa vào lịch sử
            print("🔍 Dr.Lc đang phân tích ngữ cảnh câu hỏi...")
            standalone_query = condense_query(user_query, chat_history)
            
            if standalone_query != user_query:
                print(f"   [Debug: Câu hỏi đã được làm rõ thành -> '{standalone_query}']")
            
            # 2. Retrieval: Tìm kiếm các chunk liên quan bằng Standalone Query
            print("🤖 Dr.Lc đang truy xuất thông tin thuốc...")
            chunks = retrieve_chunks(standalone_query, embed_model, collection, k=4)
            
            # Kiểm tra xem có lấy được gì không
            if not chunks or chunks[0]["similarity"] < 0.15:
                print("🤖 Dr.Lc: Tôi xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi trong dữ liệu của nhà thuốc Long Châu.")
                continue
            
            print(f"   [Debug: Đã tìm thấy {len(chunks)} chunks. Độ tương đồng cao nhất: {chunks[0]['similarity']:.3f}]")
            print("🤖 Dr.Lc đang suy nghĩ trả lời (chạy local)...")
            print("\n🤖 Dr.Lc: ", end="", flush=True)
            
            full_response = []
            
            # 3. Generation & Streaming: Gọi API chat Ollama
            # Chú ý: Ở đây ta vẫn truyền `user_query` thô cho LLM chat, vì LLM chat sẽ đọc toàn bộ `chat_history`.
            # Chúng ta chỉ dùng `standalone_query` để đi tìm kiếm vector chính xác ở bước trên!
            for token in generate_chat_stream(user_query, chunks, chat_history):
                print(token, end="", flush=True)
                full_response.append(token)
                
            full_text = "".join(full_response)
            
            # 4. Lưu lượt chat hiện tại vào lịch sử
            # Lưu user_query thô để giữ đúng hội thoại tự nhiên của người dùng
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": full_text})
            
            # 5. Đính kèm Nguồn tham khảo (chỉ khi không phải hỏi triệu chứng chung và không lỗi)
            is_symptom_query = any(kw in standalone_query.lower() for kw in symptom_keywords)
            if not is_symptom_query and "Tôi xin lỗi" not in full_text and "Lỗi" not in full_text:
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
            
            print() # Xuống dòng
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt bạn!")
            break
        except Exception as e:
            print(f"\n❌ LỖI HỆ THỐNG: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()
