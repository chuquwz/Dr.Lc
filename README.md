# Dr.Lc - Tro Ly Duoc Si Ao (Local Chat-RAG System)

Dr.Lc (Doctor Long Chau) la mot he thong chatbot RAG (Retrieval-Augmented Generation) chay hoan toan ngoai tuyen (100% Local), duoc xay dung de tu van y te, huong dan su dung va tra cuu thong tin thuoc dua tren co so du lieu thuc te tu he thong nha thuoc Long Chau.

Du an nay duoc thiet ke de giai quyet cac van de thuc te cua RAG nhu: toi uu chi phi API, bao mat du lieu y te, kiem soat hien tuong ao giac (hallucination) cua LLM va xu ly truy van hoi thoai thong minh co ngu canh.

---

## Kien Truc He Thong (System Architecture)

He thong duoc thiet ke theo mo hinh Two-Stage RAG Pipeline chay hoan toan local:

```text
               +----------------------------------------+
               |  Du lieu tho (1,019 HTML crawl tu LC)  |
               +-------------------+--------------------+
                                   |
                       (parse_html.py & chunking.py)
                                   v
                 +-----------------------------------+
                 | 11,153 Chunks JSON co Metadata    |
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

## Diem Sang Ky Thuat (Technical Highlights)

* **Hybrid Retrieval (Dense + Sparse):** Su ket hop hoan hao giua tim kiem ngu nghia (Dense Retrieval bang ChromaDB + mo hinh local paraphrase-multilingual-MiniLM-L12-v2) va tim kiem tu khoa y te chinh xac (Sparse Retrieval bang thuat toan BM25Okapi local).
* **RRF (Reciprocal Rank Fusion):** Su dung thuat toan quan hoa toan hoc de gop va tai xep hang ket qua tu hai luong tim kiem, dam bao cac tai lieu khop tu khoa doc nhat (ten thuoc, hoat chat) va dung trieu chung luon duoc dua len dau ngu canh.
* **Conversation Memory & Query Condensing:**
  * Tich hop bo nho truot luu giu toi da 5 luot hoi thoai gan nhat.
  * Su dung LLM local phan tich lich su de viet lai cau hoi tho phu thuoc ngu canh (vi du: "ke don cho toi", "no gia bao nhieu?") thanh cau hoi tim kiem doc lap day du nghia (Standalone Query) truoc khi dua vao VectorDB.
* **Double-Layer Guardrails (Hang rao bao ve 2 lop):**
  * **Lop 1 (LLM Semantic):** Tu dong nhan dien va tu choi cac cau hoi khong lien quan den y te, suc khoe (thoi tiet, toan hoc, viet code...) bang cau xin loi tieu chuan.
  * **Lop 2 (Post-processing):** Tu dong phat hien phan hoi tu choi de ngan hien thi nguon san pham tham khao rac.
* **Real-time Streaming & Light Web UI:** 
  * Ho tro che do stream chu thoi gian thuc tren phien ban console.
  * Giao dien Web UI tinh gian, phong sang, khong hieu ung chuyen dong ruom ra, tai trang sieu nhe va phan hoi tuc thi.

---

## Cau Truc Thu Muc Du An (Project Structure)

```text
Dr.Lc/
│
├── data-raw/                  # Thu muc chua 1,019 file HTML thu da crawl
├── data-processed/            # Du lieu da qua xu ly va file chunks.json
├── vector-store/              # Co so du lieu vector ChromaDB cuc bo
│
├── src/
│   ├── data_processing/
│   │   ├── parse_html.py      # Doc va lam sach Next.js JSON tu file HTML
│   │   └── chunking.py        # Chia nho thong tin thuoc kem metadata
│   │
│   ├── embedding/
│   │   └── embed_chunks.py    # Tao embedding bang transformer va nap vao database
│   │
│   ├── chatbot/
│   │   └── rag_chatbot.py     # Chatbot RAG chay tren Console (giao dien terminal)
│   │
│   ├── templates/
│   │   └── index.html         # Giao dien Web UI tinh gian phong sang (duoi 5KB)
│   │
│   └── app.py                 # Backend API va Web Server dung Flask
│
├── requirements.txt           # Danh sach cac thu vien su dung cua du an
└── README.md                  # Tai lieu dac ta du an
```

---

## Cong Nghe Su Dung (Tech Stack)

* **Language:** Python 3.10+
* **Web Server:** Flask (Lightweight API)
* **Vector Database:** ChromaDB (Local persistent)
* **Embedding Model:** sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dimensions - local)
* **Sparse Search:** rank-bm25 (Okapi BM25 local)
* **Local LLM Engine:** Ollama (localhost:11434)
* **Primary LLM Model:** qwen3:4b hoac qwen2.5:3b
* **Frontend:** Vanilla HTML, CSS, JavaScript (Sie nhe, khong dung library ngoai)

---

## Huong Dan Cai Dat & Chay He Thong

### 1. Chuan Bi Moi Truong
* Cai dat Python 3.10 tro len.
* Tai va cai dat Ollama tu ollama.com.
* Tai mo hinh Qwen local ve Ollama qua cmd:
  ```bash
  ollama pull qwen2.5:3b
  ```

### 2. Cai Dat Thu Vien Dependency
Clone du an ve may, mo terminal tai thu muc goc va chay:
```bash
pip install -r requirements.txt
```

### 3. Chuan Bi Du Lieu & Nap Vector Database
*(Dam bao thu muc data-raw chua cac file HTML tho da crawl tu Long Chau)*

1. **Parse du lieu HTML sang JSON:**
   ```bash
   python src/data_processing/parse_html.py
   ```
2. **Cat nho du lieu thanh Chunks:**
   ```bash
   python src/data_processing/chunking.py
   ```
3. **Embed va nap du lieu vao ChromaDB local:**
   ```bash
   python src/embedding/embed_chunks.py
   ```

---

## Huong Dan Su Dung (How to Run)

### Lua chon 1: Chay giao dien Web UI (Khuyen dung)
1. Khoi chay Web Server va Backend API:
   ```bash
   python src/app.py
   ```
2. Mo trinh duyet web bat ky va truy cap:
   👉 **http://127.0.0.1:5000**
3. Su dung giao dien chat phong sang de tu van suc khoe va thong tin thuoc.

### Lua chon 2: Chay giao dien Console (Terminal)
1. Khoi chay ung dung console truc tiep:
   ```bash
   python src/chatbot/rag_chatbot.py
   ```
2. Tuong tac qua terminal. Go `exit` hoac `quit` de thoat.

---

## Huong Phat Trien Tuong Lai (Roadmap)

* [ ] **Two-stage Retrieval voi Cross-Encoder Reranker:** Trien khai them mo hinh amberyouying/bge-reranker-base-multilingual local de tinh chinh thu hang ngu canh truoc khi dua vao LLM.
* [ ] **Evaluation Pipeline:** Su dung framework Ragas de danh gia dinh luong do chinh xac (faithfulness, answer relevance, context recall) cua chatbot.
