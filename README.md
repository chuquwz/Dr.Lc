# 🏥 Dr.Lc — Trợ Lý Dược Sĩ Ảo (Local Chat-RAG System)

**Dr.Lc (Doctor Long Châu)** là một hệ thống chatbot RAG (Retrieval-Augmented Generation) chạy hoàn toàn ngoại tuyến (100% Local), được xây dựng để tư vấn y tế, hướng dẫn sử dụng và tra cứu thông tin thuốc dựa trên cơ sở dữ liệu thực tế từ hệ thống nhà thuốc Long Châu.

Dự án này được thiết kế để giải quyết các vấn đề thực tế của RAG như: tối ưu chi phí API, bảo mật dữ liệu y tế, kiểm soát hiện tượng ảo giác (hallucination) của LLM và xử lý truy vấn hội thoại thông minh có ngữ cảnh.

---

## 🛠️ Kiến Trúc Hệ Thống (System Architecture)

Hệ thống được thiết kế theo mô hình **Two-Stage RAG Pipeline** chạy hoàn toàn local:

```text
               +----------------------------------------+
               |  Dữ liệu thô (1,019 HTML crawl từ LC)  |
               +-------------------+--------------------+
                                   |
                       (parse_html.py & chunking.py)
                                   v
                 +-----------------------------------+
                 | 11,153 Chunks JSON có Metadata    |
                 +-----------------+-----------------+
                                   |
                   +---------------+---------------+
                   | (embed_chunks.py)             | (BM25 Indexing)
                   v                               v
         +-------------------+           +-------------------+
         |  ChromaDB (Dense) |           |    BM25 (Sparse)  |
         +---------+---------+           +---------+---------+
                   |                               |
                   +---------------+---------------+
                                   |
                        (User Query / Chat History)
                                   v
                         (condense_query LLM)
                                   v
                      [ Standalone Search Query ]
                                   v
                   +---------------+---------------+
                   |                               |
              (Dense Search)                 (Sparse Search)
                   |                               |
                   v                               v
             [ Top-20 Vector ]               [ Top-20 Keyword ]
                   |                               |
                   +---------------+---------------+
                                   |
                     (Reciprocal Rank Fusion - RRF)
                                   v
                             [ Top-4 Chunks ]
                                   v
                   (Guardrails & System Instruction)
                                   v
                         (Ollama API /api/chat)
                                   v
                  [ Streaming Response to Console ]
```

---

## ✨ Điểm Sáng Kỹ Thuật (Technical Highlights)

* **Hybrid Retrieval (Dense + Sparse):** Sự kết hợp hoàn hảo giữa tìm kiếm ngữ nghĩa (Dense Retrieval bằng ChromaDB + mô hình local `paraphrase-multilingual-MiniLM-L12-v2`) và tìm kiếm từ khóa y tế chính xác (Sparse Retrieval bằng thuật toán `BM25Okapi` local).
* **RRF (Reciprocal Rank Fusion):** Sử dụng thuật toán chuẩn hóa toán học để gộp và tái xếp hạng kết quả từ hai luồng tìm kiếm, đảm bảo các tài liệu khớp từ khóa độc nhất (tên thuốc, hoạt chất) và đúng triệu chứng luôn được đưa lên đầu ngữ cảnh.
* **Conversation Memory & Query Condensing:** 
  * Tích hợp bộ nhớ trượt lưu giữ tối đa 5 lượt hội thoại gần nhất.
  * Sử dụng LLM local phân tích lịch sử để viết lại câu hỏi thô phụ thuộc ngữ cảnh (ví dụ: *"kê đơn cho tôi"*, *"nó giá bao nhiêu?"*) thành câu hỏi tìm kiếm độc lập đầy đủ nghĩa (**Standalone Query**) trước khi đưa vào VectorDB.
* **Double-Layer Guardrails (Hàng rào bảo vệ 2 lớp):**
  * **Lớp 1 (LLM Semantic):** Tự động nhận diện và từ chối các câu hỏi không liên quan đến y tế, sức khỏe (thời tiết, toán học, viết code...) bằng câu xin lỗi tiêu chuẩn.
  * **Lớp 2 (Post-processing):** Tự động phát hiện phản hồi từ chối để ngăn hiển thị nguồn sản phẩm tham khảo rác.
* **Real-time Streaming:** Sử dụng kết nối streaming (`stream=True`) từ API Ollama để phản hồi chữ chạy thời gian thực trên console, giải quyết triệt để lỗi nghẽn và timeout của requests khi chạy LLM local trên CPU.

---

## 💻 Công Nghệ Sử Dụng (Tech Stack)

* **Language:** Python 3.10+
* **Vector Database:** ChromaDB (Local persistent)
* **Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions - local)
* **Sparse Search:** `rank-bm25` (Okapi BM25 local)
* **Local LLM Engine:** Ollama (localhost:11434)
* **Primary LLM Model:** `qwen3:4b` hoặc `qwen2.5:3b` (mô hình tối ưu cực tốt cho tiếng Việt)
* **Libraries:** `requests`, `torch`, `beautifulsoup4`, `python-dotenv`

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Hệ Thống

### 1. Chuẩn Bị Môi Trường
* Cài đặt **Python 3.10** trở lên.
* Tải và cài đặt **Ollama** từ [ollama.com](https://ollama.com).
* Tải mô hình Qwen local về Ollama qua cmd:
  ```bash
  ollama pull qwen2.5:3b
  # Hoặc mô hình qwen3:4b nếu có sẵn
  ```

### 2. Cài Đặt Thư Viện Dependency
Clone dự án về máy, mở terminal tại thư mục gốc và chạy:
```bash
pip install -r requirements.txt
```

### 3. Chuẩn Bị Dữ Liệu & Nạp Vector Database
*(Đảm bảo thư mục `data-raw` chứa các file HTML thô đã crawl từ Long Châu)*

1. **Parse dữ liệu HTML sang JSON:**
   ```bash
   python src/data_processing/parse_html.py
   ```
2. **Cắt nhỏ dữ liệu thành Chunks:**
   ```bash
   python src/data_processing/chunking.py
   ```
3. **Embed và nạp dữ liệu vào ChromaDB local:**
   ```bash
   python src/embedding/embed_chunks.py
   ```
   *(Tiến trình này chạy local mất khoảng 3 - 5 phút tùy thuộc cấu hình máy của bạn, cơ sở dữ liệu sẽ được lưu tại thư mục `vector-store/`)*

### 4. Khởi Chạy Chatbot
Đảm bảo Ollama đang chạy trên máy của bạn, sau đó khởi động console chatbot:
```bash
python src/chatbot/rag_chatbot.py
```

---

## 🛣️ Hướng Phát Triển Tương Lai (Roadmap)

* [ ] **Two-stage Retrieval với Cross-Encoder Reranker:** Triển khai thêm mô hình `amberyouying/bge-reranker-base-multilingual` local để tinh chỉnh thứ hạng ngữ cảnh trước khi đưa vào LLM.
* [ ] **Web UI Dashboard:** Xây dựng giao diện web đẹp mắt bằng **Streamlit** hoặc **Next.js** thay vì giao diện terminal console hiện tại.
* [ ] **Evaluation Pipeline:** Sử dụng framework Ragas để đánh giá định lượng độ chính xác (faithfulness, answer relevance, context recall) của chatbot.
