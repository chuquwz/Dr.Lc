import sys
import os
from flask import Flask, request, jsonify, render_template

# Thêm PROJECT_ROOT vào Python path để import chính xác module RAG
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.chatbot.rag_chatbot import init_components, condense_query, hybrid_retrieve, generate_chat_stream

# Khởi tạo ứng dụng Flask
app = Flask(__name__, template_folder="templates")

# Khởi tạo các thành phần RAG duy nhất một lần khi máy chủ khởi động
print("Dang khoi dong Dr.Lc RAG Backend Server...")
embed_model, collection, all_chunks, bm25_model = init_components()
print("May chu Dr.Lc RAG da san sang phuc vu tai http://127.0.0.1:5000")


@app.route('/')
def home():
    """ Serve trang giao diện chính Web UI """
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """ API nhận tin nhắn từ giao diện web, chạy RAG local và trả về JSON """
    try:
        data = request.json or {}
        message = data.get('message', '').strip()
        history = data.get('history', [])
        
        if not message:
            return jsonify({"error": "Nội dung câu hỏi không được trống"}), 400
            
        # 1. Condense Query: Viết lại câu hỏi độc lập dựa trên lịch sử
        standalone_query = condense_query(message, history)
        
        # 2. Retrieval: Tìm kiếm Hybrid (Vector + BM25) gộp RRF
        chunks = hybrid_retrieve(standalone_query, embed_model, collection, all_chunks, bm25_model, k=4)
        
        # 3. Generation: Gọi sinh câu trả lời (Gom toàn bộ các token stream thành văn bản hoàn chỉnh)
        answer_parts = []
        for token in generate_chat_stream(message, chunks, history):
            answer_parts.append(token)
        answer = "".join(answer_parts)
        
        # 4. Trích xuất nguồn liên kết sản phẩm (đối với câu hỏi thuốc cụ thể)
        sources = []
        symptom_keywords = ["là dấu hiệu", "là bệnh gì", "bị bệnh gì", "nguyên nhân", "dấu hiệu của", "triệu chứng của"]
        is_symptom = any(kw in standalone_query.lower() for kw in symptom_keywords)
        is_out_of_scope = "nằm ngoài phạm vi hỗ trợ" in answer
        
        if not is_symptom and not is_out_of_scope and "Tôi xin lỗi" not in answer and "Lỗi" not in answer:
            seen_urls = set()
            for chunk in chunks:
                meta = chunk["metadata"]
                p_name = meta.get("product_name", "Thuốc")
                p_url = meta.get("product_url", "")
                if p_name and p_url and p_url not in seen_urls:
                    seen_urls.add(p_url)
                    short_name = p_name.split("(")[0].strip() if "(" in p_name else p_name
                    sources.append({"name": short_name, "url": p_url})
                    
        return jsonify({
            "answer": answer,
            "sources": sources
        })
        
    except Exception as e:
        return jsonify({"error": f"Lỗi hệ thống: {str(e)}"}), 500


if __name__ == '__main__':
    # Chạy cục bộ trên cổng 5000, tắt debug để tránh nạp lại mô hình 2 lần
    app.run(host='127.0.0.1', port=5000, debug=False)
