# Dr.Lc - Trợ Lý Dược Sĩ Ảo (Local Chat-RAG System)

Dr.Lc (Doctor Long Châu) là một hệ thống chatbot RAG (Retrieval-Augmented Generation) chạy hoàn toàn ngoại tuyến (100% Local), được xây dựng để tư vấn y tế, hướng dẫn sử dụng và tra cứu thông tin thuốc dựa trên cơ sở dữ liệu thực tế từ hệ thống nhà thuốc Long Châu.

Dự án này được thiết kế để giải quyết các vấn đề thực tế của RAG như: tối ưu chi phí API, bảo mật dữ liệu y tế, kiểm soát hiện tượng ảo giác (hallucination) của LLM và xử lý truy vấn hội thoại thông minh có ngữ cảnh.

---

## Kiến Trúc Hệ Thống (System Architecture)

Hệ thống được thiết kế theo mô hình Two-Stage RAG Pipeline chạy hoàn toàn local:

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

## Điểm Sáng Kỹ Thuật (Technical Highlights)

* **Hybrid Retrieval (Dense + Sparse):** Sự kết hợp hoàn hảo giữa tìm kiếm ngữ nghĩa (Dense Retrieval bằng ChromaDB + mô hình local paraphrase-multilingual-MiniLM-L12-v2) và tìm kiếm từ khóa y tế chính xác (Sparse Retrieval bằng thuật toán BM25Okapi local).
* **RRF (Reciprocal Rank Fusion):** Sử dụng thuật toán chuẩn hóa toán học để gộp và tái xếp hạng kết quả từ hai luồng tìm kiếm, đảm bảo các tài liệu khớp từ khóa độc nhất (tên thuốc, hoạt chất) và đúng triệu chứng luôn được đưa lên đầu ngữ cảnh.
* **Conversation Memory & Query Condensing:**
  * Tích hợp bộ nhớ trượt lưu giữ tối đa 5 lượt hội thoại gần nhất.
  * Sử dụng LLM local phân tích lịch sử để viết lại câu hỏi thô phụ thuộc ngữ cảnh (ví dụ: "kê đơn cho tôi", "nó giá bao nhiêu?") thành câu hỏi tìm kiếm độc lập đầy đủ nghĩa (Standalone Query) trước khi đưa vào VectorDB.
* **Double-Layer Guardrails (Hàng rào bảo vệ 2 lớp):**
  * **Lớp 1 (LLM Semantic):** Tự động nhận diện và từ chối các câu hỏi không liên quan đến y tế, sức khỏe (thời tiết, công nghệ, toán học, viết code...) bằng câu xin lỗi tiêu chuẩn.
  * **Lớp 2 (Post-processing):** Tự động phát hiện phản hồi từ chối để ngăn hiển thị nguồn sản phẩm tham khảo rác.
* **Real-time Streaming & Light Web UI:** 
  * Hỗ trợ chế độ stream chữ thời gian thực trên phiên bản console.
  * Giao diện Web UI tinh giản, phông sáng, không hiệu ứng chuyển động rườm rà, tải trang siêu nhẹ và phản hồi tức thì.

---

## Cấu Trúc Thư Mục Dự Án (Project Structure)

```text
Dr.Lc/
│
├── data-raw/                  # Thư mục chứa 1,019 file HTML thô đã crawl
├── data-processed/            # Dữ liệu đã qua xử lý và file chunks.json
├── vector-store/              # Cơ sở dữ liệu vector ChromaDB cục bộ
│
├── src/
│   ├── data_processing/
│   │   ├── parse_html.py      # Đọc và làm sạch Next.js JSON từ file HTML
│   │   └── chunking.py        # Chia nhỏ thông tin thuốc kèm metadata
│   │
│   ├── embedding/
│   │   └── embed_chunks.py    # Tạo embedding bằng transformer và nạp vào database
│   │
│   ├── chatbot/
│   │   └── rag_chatbot.py     # Chatbot RAG chạy trên Console (giao diện terminal)
│   │
│   ├── templates/
│   │   └── index.html         # Giao diện Web UI tinh giản phông sáng (dưới 5KB)
│   │
│   └── app.py                 # Backend API và Web Server dùng Flask
│
├── requirements.txt           # Danh sách các thư viện sử dụng của dự án
└── README.md                  # Tài liệu đặc tả dự án
```

---

## Công Nghệ Sử Dụng (Tech Stack)

* **Language:** Python 3.10+
* **Web Server:** Flask (Lightweight API)
* **Vector Database:** ChromaDB (Local persistent)
* **Embedding Model:** sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions - local)
* **Sparse Search:** rank-bm25 (Okapi BM25 local)
* **Local LLM Engine:** Ollama (localhost:11434)
* **Primary LLM Model:** qwen3:4b hoặc qwen2.5:3b
* **Frontend:** Vanilla HTML, CSS, JavaScript (Siêu nhẹ, không dùng thư viện ngoài)

---

## Hướng Dẫn Cài Đặt & Chạy Hệ Thống

### 1. Chuẩn Bị Môi Trường
* Cài đặt Python 3.10 trở lên.
* Tải và cài đặt Ollama từ ollama.com.
* Tải mô hình Qwen local về Ollama qua cmd:
  ```bash
  ollama pull qwen2.5:3b
  ```

### 2. Cài Đặt Thư Viện Dependency
Clone dự án về máy, mở terminal tại thư mục gốc và chạy:
```bash
pip install -r requirements.txt
```

### 3. Chuẩn Bị Dữ Liệu & Nạp Vector Database
*(Đảm bảo thư mục data-raw chứa các file HTML thô đã crawl từ Long Châu)*

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

---

## Hướng Dẫn Sử Dụng (How to Run)

### Lựa chọn 1: Chạy giao diện Web UI (Khuyên dùng)
1. Khởi chạy Web Server và Backend API:
   ```bash
   python src/app.py
   ```
2. Mở trình duyệt web bất kỳ và truy cập:
   👉 **http://127.0.0.1:5000**
3. Sử dụng giao diện chat phông sáng để tư vấn sức khỏe và thông tin thuốc.

### Lựa chọn 2: Chạy giao diện Console (Terminal)
1. Khởi chạy ứng dụng console trực tiếp:
   ```bash
   python src/chatbot/rag_chatbot.py
   ```
2. Tương tác qua terminal. Gõ `exit` hoặc `quit` để thoát.

---

## Hướng Phát Triển Tương Lai (Roadmap)

* [ ] **Two-stage Retrieval với Cross-Encoder Reranker:** Triển khai thêm mô hình amberyouying/bge-reranker-base-multilingual local để tinh chỉnh thứ hạng ngữ cảnh trước khi đưa vào LLM.
* [ ] **Evaluation Pipeline:** Sử dụng framework Ragas để đánh giá định lượng độ chính xác (faithfulness, answer relevance, context recall) của chatbot.
