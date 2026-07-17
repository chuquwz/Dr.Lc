"""
Dr.Lc RAG Chatbot - Phien ban Chat-RAG hoan chinh (Hybrid Retrieval + RRF + Lich su Hoi thoai)
========================================================================================
Script nay thuc hien su ket hop nang cao nhat cua pipeline RAG chay Local (Khong Reranker):
1. Nhan cau hoi tu nguoi dung qua terminal.
2. Dung LLM local (Ollama) viet lai cau hoi tho thanh Standalone Query (cau hoi doc lap) dua vao Lich su Hoi thoai.
3. Chay song song 2 luong truy xuat bang Standalone Query:
   - Dense Search (Vector): Dung SentenceTransformer + ChromaDB de lay Top-20 chunks theo nguoi nghia.
   - Sparse Search (Tu khoa): Dung thuat toan BM25Okapi local de lay Top-20 chunks khop tu khoa chinh xac.
4. Gop va xep hang lai ket qua bang thuat toan RRF (Reciprocal Rank Fusion) voi hang so K=60.
5. Lay Top-4 chunks co diem RRF cao nhat lam nguoi canh dang tin cay.
6. Xay dung prompt y te an toan va goi API /api/chat cua Ollama o che do Streaming de tra loi thoi gian thuc.
7. Hien thi chu chay, in nguon tham khao y khoa va cap nhat lich su chat.

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

# Fix encoding cho Windows console (tieng Viet khong loi)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# ============================================================
# CAU HINH & LOAD MOI TRUONG
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CHUNKS_FILE = os.path.join(PROJECT_ROOT, "data-processed", "chunks.json")
VECTOR_STORE_DIR = os.path.join(PROJECT_ROOT, "vector-store")
COLLECTION_NAME = "dr_lc_products"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Ollama API URL cho Chat va Generate
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
LOCAL_LLM_MODEL = "qwen3:4b"

# So luong luot hoi thoai toi da duoc luu trong bo nho (tranh qua tai ngu canh)
MAX_HISTORY_TURNS = 5


# ============================================================
# HAM BO TRO: TOKENIZER CHO TIENG VIET (DUNG CHO BM25)
# ============================================================

def clean_and_tokenize(text: str) -> list[str]:
    """
    Tach tu (tokenizer) don gian cho tieng Viet de dung cho BM25.
    """
    cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = [token for token in cleaned_text.split() if token.strip()]
    return tokens


# ============================================================
# KHOI TAO MO HINH & DATABASE
# ============================================================

def init_components():
    """
    Khoi tao dong thoi:
    1. Kiem tra ket noi toi Ollama local
    2. Load mo hinh embedding local (SentenceTransformer)
    3. Ket noi toi ChromaDB local (Dense Search)
    4. Load file chunks.json va build chi muc BM25 (Sparse Search)
    """
    # 1. Kiem tra ket noi Ollama
    try:
        response = requests.get("http://localhost:11434/", timeout=2)
        if response.status_code == 200:
            print("Da ket noi thanh cong toi Ollama (localhost:11434)")
    except requests.exceptions.ConnectionError:
        print("LOI: Khong the ket noi toi Ollama!")
        print("   Vui long mo ung dung Ollama tren may cua ban truoc.")
        print("   Hoac chay lenh 'ollama serve' trong cmd.")
        sys.exit(1)
        
    # 2. Khoi tao Embedding Model Local
    print("Dang load mo hinh embedding local (SentenceTransformer)...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    
    # 3. Khoi tao ChromaDB
    print("Dang ket noi toi Vector Database (ChromaDB)...")
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"LOI: Khong tim thay collection '{COLLECTION_NAME}' trong ChromaDB!")
        print("   Vui long chay script 'src/embedding/embed_chunks.py' truoc de nap du lieu.")
        sys.exit(1)
        
    # 4. Load chunks va xay dung chi muc BM25
    print(f"Dang doc file chunks va xay dung chi muc BM25...")
    start_time = time.time()
    
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
        
    # Tokenize toan bo corpus chunks
    tokenized_corpus = [clean_and_tokenize(chunk["content"]) for chunk in all_chunks]
    
    # Khoi tao BM25Okapi (thuat toan BM25 chuan)
    bm25_model = BM25Okapi(tokenized_corpus)
    
    print(f"Da dung xong chi muc BM25 cho {len(all_chunks)} chunks trong {time.time() - start_time:.2f} giay.")
    return embed_model, collection, all_chunks, bm25_model


# ============================================================
# BUOC 1: CONVERSE & CONDENSE QUERY (TOM TAT & VIET LAI CAU HOI)
# ============================================================

def condense_query(raw_query: str, chat_history: list[dict]) -> str:
    """
    Dua vao lich su chat va cau hoi tho moi, yeu cau LLM viet lai thanh 
    mot cau tim kiem doc lap (Standalone Query) chua day du ngu canh cu.
    """
    if not chat_history:
        return raw_query
        
    history_str = ""
    for msg in chat_history[-MAX_HISTORY_TURNS*2:]:
        role_label = "Nguoi dung" if msg["role"] == "user" else "Tro ly Duoc si"
        history_str += f"{role_label}: {msg['content']}\n"
        
    system_prompt = (
        "Nhiem vu cua ban la doc LICH SU HOI THOAI va CAU HOI MOI duoi day, sau do viet lai cau hoi "
        "thanh mot cau hoi tim kiem DOC LAP, DAY DU Y NGHIA (khong dung tu thay the mo ho nhu 'no', 'chung', 'cho toi', 'nhu the nao').\n"
        "Chi tra ve cau hoi da duoc viet lai bang tieng Viet, khong giai thich gi them."
    )
    
    prompt = (
        f"LICH SU HOI THOAI:\n"
        f"{history_str}\n"
        f"CAU HOI MOI: {raw_query}\n\n"
        f"CAU HOI DOC LAP SAU KHI VIET LAI:"
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
# BUOC 2: HYBRID RETRIEVAL & RRF (TRUY XUAT HON HOP)
# ============================================================

def hybrid_retrieve(query: str, embed_model, collection, all_chunks, bm25_model, k: int = 4) -> list[dict]:
    """
    Thuc hien Hybrid Retrieval ket hop Dense (Vector) + Sparse (BM25) su dung RRF.
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

    # ── 4. SAP XEP & LOC TOP-K KET QUA ────────────────────────
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
# BUOC 3: GENERATION (GOI /API/CHAT OLLAMA DANG STREAMING)
# ============================================================

def generate_chat_stream(query: str, retrieved_chunks: list[dict], chat_history: list[dict]):
    """
    Su dung endpoint `/api/chat` de duy tri lich su hoi thoai.
    Tich hop prompt y te va ngu canh y khoa vao tin nhan hien tai.
    """
    context_items = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        p_name = meta.get("product_name", "Thuoc")
        context_items.append(
            f"--- TAI LIEU {i} (San pham: {p_name} | Nguon tim thay: {chunk['source_type']}) ---\n"
            f"{chunk['content']}\n"
        )
    context_str = "\n".join(context_items)
    
    system_instruction = (
        "Ban la Dr.Lc (Doctor Long Chau) - mot tro ly duoc si ao thong minh chuyen tu van y te "
        "va ho tro mua thuoc dua tren du lieu san pham cua nha thuoc Long Chau.\n\n"
        "Nhiem vu cua ban:\n"
        "1. Tra loi cau hoi cua nguoi dung mot cach chinh xac, chuyen nghiep, de hieu bang tieng Viet.\n"
        "2. PHAN LOAI CAU HOI & XU LY NGUON TRI THUC:\n"
        "   - THE A (Hoi ve Thuoc - Vi du: lieu dung, tac dung phu, gia ca, cach uong cua mot thuoc cu the): Ban BAT BUOC chi su dung thong tin "
        "duoc cung cap trong phan 'NGUOC CANH TRUY XUAT' duoi day. Tuyet doi khong tu y bia thong tin thuoc nam ngoai ngu canh.\n"
        "   - THE B (Hoi ve Trieu chung, Benh hoc hoac Loi khuyen suc khoe - Vi du: dau bung la benh gi, toi nen lam gi): "
        "Vi phan 'NGUOC CANH TRUY XUAT' chi chua thong tin thuoc, ban DUOC PHEP su dung tri thuc y khoa rong lon cua minh de giai thich "
        "cac nguyen nhan pho bien (nhu thieu ngu, day hoi, huyet ap...) va dua ra loi khuyen loi song. TUYET DOI khong khuyen dung "
        "cac thuoc nam ngoai Nguoc canh truy xuat, chi duoc tu van huong di kham.\n"
        "   - THE C (Hoi ngoai pham vi - Khong lien quan den y te, suc khoe, benh tat, loi song lanh manh hay thong tin thuoc. Vi du: thoi tiet, cong nghe, toan hoc, viet code, dich thuat, tho ca...): "
        "Ban BAT BUOC phai tu choi tra loi mot cach lich su va noi chinh xac cau sau: 'Toi xin loi, cau hoi nay nam ngoai pham vi ho tro cua tro ly y te Dr.Lc. Toi chi co the tu van cac van de lien quan den y te, suc khoe, benh tat va thong tin thuoc.'\n"
        "3. Neu cau hoi thuoc THE A nhung NGUOC CANH khong chua thong tin, hay tra loi ro la: "
        "'Toi xin loi, thong tin ve san pham nay hien khong co trong du lieu cua nha thuoc Long Chau. "
        "De dam bao an toan, ban nen tham khao y kien cua bac si hoac duoc si chuyen mon.'\n"
        "4. CANH BAO AN TOAN: Khi tu van y te, luon khuyen nguoi dung di kham bac si de duoc chan doan chinh xac."
    )
    
    messages = []
    messages.append({"role": "system", "content": system_instruction})
    messages.extend(chat_history[-MAX_HISTORY_TURNS*2:])
    
    current_content = (
        f"NGUOC CANH TRUY XUAT:\n"
        f"{context_str}\n\n"
        f"CAU HOI HIEN TAI:\n"
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
                f"Loi Ollama (HTTP {response.status_code}): {response.text}\n\n"
                f"Goi y khac phuc: Doi model lai thanh mot model nhe hon hoac khoi dong lai Ollama."
            )
            return
            
        for line in response.iter_lines():
            if line:
                res_json = json.loads(line.decode('utf-8'))
                
                if "error" in res_json:
                    yield f"Loi tu Ollama API: {res_json['error']}"
                    return
                
                message_chunk = res_json.get("message", {})
                token = message_chunk.get("content", "")
                yield token
                
                if res_json.get("done", False):
                    break
                    
    except requests.exceptions.Timeout:
        yield "Loi: Thoi gian phan hoi tu Ollama qua han (Timeout). Ban hay thu lai."
    except Exception as e:
        yield f"Loi ket noi toi Ollama: {e}"


# ============================================================
# CHUONG TRINH CHINH (GIAO DIEN CONSOLE)
# ============================================================

def main():
    print("=" * 70)
    print("DR.LC CHATBOT - PHIEN BAN HYBRID RETRIEVAL (DENSE + SPARSE) + RRF")
    print("=" * 70)
    
    # Khoi tao ca thanh phan
    embed_model, collection, all_chunks, bm25_model = init_components()
    
    print(f"\nSystem: Dang su dung mo hinh local: '{LOCAL_LLM_MODEL}' qua Ollama.")
    print("System: Cong nghe tim kiem: Hybrid Search (ChromaDB Vector + BM25 Local) + Xep hang RRF.")
    print("System: Go 'exit' hoac 'quit' de thoat chuong trinh.")
    print("-" * 70)
    
    chat_history = []
    symptom_keywords = ["la dau hieu", "la benh gi", "bi benh gi", "nguyen nhan", "dau hieu cua", "trieu chung cua", "là dấu hiệu", "là bệnh gì", "bị bệnh gì", "nguyên nhân", "dấu hiệu của", "triệu chứng của"]
    
    while True:
        try:
            user_query = input("\nBan: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit"]:
                print("Tam biet ban! Chuc ban nhieu suc khoe!")
                break
                
            print("Dr.Lc dang phan tich nguoc canh cau hoi...")
            standalone_query = condense_query(user_query, chat_history)
            
            if standalone_query != user_query:
                print(f"   [Debug: Cau hoi da duoc lam ro thanh -> '{standalone_query}']")
            
            print("Dr.Lc dang truy xuat thong tin (Hybrid + RRF)...")
            chunks = hybrid_retrieve(standalone_query, embed_model, collection, all_chunks, bm25_model, k=4)
            
            if not chunks:
                print("Dr.Lc: Toi xin loi, toi khong tim thay thong tin lien quan den cau hoi trong du lieu cua nha thuoc Long Chau.")
                continue
            
            print("   [Debug: Top Chunks tim thay bang RRF]")
            for i, c in enumerate(chunks):
                print(f"     {i+1}. Score: {c['rrf_score']:.4f} | Nguon: {c['source_type']:6s} | {c['metadata']['product_name'][:40]}... ({c['metadata']['section']})")
            
            print("Dr.Lc dang suy nghi tra loi (chay local)...")
            print("\nDr.Lc: ", end="", flush=True)
            
            full_response = []
            for token in generate_chat_stream(user_query, chunks, chat_history):
                print(token, end="", flush=True)
                full_response.append(token)
                
            full_text = "".join(full_response)
            
            chat_history.append({"role": "user", "content": user_query})
            chat_history.append({"role": "assistant", "content": full_text})
            
            is_symptom_query = any(kw in standalone_query.lower() for kw in symptom_keywords)
            is_out_of_scope = "nam ngoai pham vi ho tro" in full_text or "nằm ngoài phạm vi hỗ trợ" in full_text
            if not is_symptom_query and not is_out_of_scope and "Toi xin loi" not in full_text and "Loi" not in full_text and "Tôi xin lỗi" not in full_text:
                sources = {}
                for chunk in chunks:
                    meta = chunk["metadata"]
                    p_name = meta.get("product_name", "Thuoc")
                    p_url = meta.get("product_url", "")
                    if p_name and p_url:
                        sources[p_name] = p_url
                
                if sources:
                    print("\n\nThong tin san pham tham khao:", end="")
                    for name, url in sources.items():
                        short_name = name.split("(")[0].strip() if "(" in name else name
                        print(f"\n- [{short_name}]({url})", end="")
            
            print()
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\nTam biet ban!")
            break
        except Exception as e:
            print(f"\nLOI HE THONG: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()
