"""
Chunking Script cho Dr.Lc RAG Chatbot
======================================
Script này thực hiện bước 2 trong pipeline RAG: chia dữ liệu đã xử lý thành
các "chunk" (đoạn nhỏ) phù hợp để embedding.

CHIẾN LƯỢC: Section-Based Chunking
- Mỗi sản phẩm thuốc có nhiều phần (mô tả, công dụng, liều dùng, tác dụng phụ...)
- Mỗi phần trở thành 1 chunk riêng biệt
- Mỗi chunk được gắn kèm metadata (tên thuốc, SKU) để giữ ngữ cảnh

TẠI SAO KHÔNG DÙNG FIXED-SIZE?
- Dữ liệu đã có cấu trúc rõ ràng theo từng mục (usage, dosage, ...)
- Cắt theo số từ sẽ phá vỡ ranh giới tự nhiên giữa các mục
- Ví dụ: chunk chứa nửa phần "Công dụng" + nửa phần "Liều dùng" = vô nghĩa

TẠI SAO GẮN TÊN THUỐC VÀO MỖI CHUNK?
- Khi tìm kiếm, mỗi chunk sống độc lập (vector DB không biết chunk nào thuộc sản phẩm nào)
- Nếu chunk chỉ ghi "Người lớn: 1 viên mỗi 6 giờ" → không biết là thuốc gì
- Phải ghi "Decolgen Forte | Liều dùng: Người lớn: 1 viên mỗi 6 giờ" → rõ ràng

Usage:
    python src/data_processing/chunking.py
"""

import sys
import io
import os
import json
import glob

# Fix encoding cho Windows console (tiếng Việt)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data-processed")
CHUNKS_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data-processed", "chunks.json")


# ============================================================
# HÀM TẠO CHUNK THÔNG TIN CHUNG
# ============================================================
def build_general_info_chunk(product: dict) -> str:
    """
    Tạo chunk "thẻ căn cước" của sản phẩm — gộp các thông tin ngắn lại với nhau.
    
    TẠI SAO GỘP?
    Vì các trường như brand, dosage_form, ingredients, price đều rất ngắn
    (chỉ vài từ mỗi trường). Nếu mỗi trường là 1 chunk riêng thì chunk
    quá bé → embedding không đủ ngữ cảnh → retrieval kém.
    
    Ví dụ output:
        "Thuốc Decolgen Forte (DECOLGEN FORTE)
         Danh mục: Thuốc > Thuốc giảm đau, hạ sốt, kháng viêm > Thuốc giảm đau hạ sốt
         Thương hiệu: United | Nước sản xuất: Việt Nam
         Dạng bào chế: Viên nén
         Thành phần (mỗi 1 viên): Paracetamol 500mg, Phenylephrine 10mg, Chlorpheniramine 2mg
         Giá: Hộp 150.000đ (Hộp 30 Vỉ x 4 Viên) | Vỉ 5.000đ | Viên 1.250đ
         Hạn sử dụng: 48 tháng
         Số đăng ký: 893100340723"
    """
    lines = []
    
    # Tên sản phẩm — dòng đầu tiên, quan trọng nhất
    name = product.get("name", "")
    official = product.get("official_name", "")
    if official and official.lower() != name.lower():
        lines.append(f"{name} ({official})")
    else:
        lines.append(name)
    
    # Danh mục — giúp chatbot hiểu thuốc này thuộc nhóm nào
    # Ví dụ: "Thuốc > Thuốc giảm đau, hạ sốt > Thuốc giảm đau hạ sốt"
    categories = product.get("categories", [])
    if categories:
        cat_names = [c["name"] for c in sorted(categories, key=lambda x: x["level"])]
        lines.append(f"Danh mục: {' > '.join(cat_names)}")
    
    # Thương hiệu & nước sản xuất
    parts = []
    if product.get("brand"):
        parts.append(f"Thương hiệu: {product['brand']}")
    if product.get("manufacturer_country"):
        parts.append(f"Nước sản xuất: {product['manufacturer_country']}")
    if parts:
        lines.append(" | ".join(parts))
    
    # Dạng bào chế (viên nén, siro, ống tiêm...)
    if product.get("dosage_form"):
        lines.append(f"Dạng bào chế: {product['dosage_form']}")
    
    # Thành phần — rất quan trọng cho việc tìm kiếm
    # User có thể hỏi: "thuốc nào có chứa Paracetamol?"
    ingredients = product.get("ingredients", [])
    if ingredients:
        ing_parts = [f"{ing['name']} {ing['amount']}" for ing in ingredients]
        per = product.get("ingredient_per", "")
        prefix = f"Thành phần (mỗi {per})" if per else "Thành phần"
        lines.append(f"{prefix}: {', '.join(ing_parts)}")
    
    # Giá — user có thể hỏi: "Decolgen Forte giá bao nhiêu?"
    prices = product.get("prices", [])
    if prices:
        price_parts = []
        for p in prices:
            if p.get("price") is not None:
                # Format giá: 150000 → "150.000"
                formatted_price = f"{p['price']:,.0f}".replace(",", ".")
                unit_str = f"{p['unit']} {formatted_price}{p.get('currency', 'đ')}"
                if p.get("specs"):
                    unit_str += f" ({p['specs']})"
                price_parts.append(unit_str)
        if price_parts:
            lines.append(f"Giá: {' | '.join(price_parts)}")
    
    # Thông tin bổ sung
    if product.get("expiration_date"):
        lines.append(f"Hạn sử dụng: {product['expiration_date']}")
    if product.get("registration_number"):
        lines.append(f"Số đăng ký: {product['registration_number']}")
    
    return "\n".join(lines)


# ============================================================
# HÀM TẠO CHUNK NỘI DUNG Y TẾ
# ============================================================
def build_medical_content_chunk(product_name: str, section_title: str, content: str) -> str:
    """
    Tạo chunk cho 1 phần nội dung y tế (công dụng, liều dùng, tác dụng phụ...).
    
    QUAN TRỌNG: Luôn gắn tên sản phẩm vào đầu chunk.
    
    Ví dụ output:
        "Thuốc Decolgen Forte — Liều dùng:
         Cách dùng: Thuốc dùng đường uống.
         Liều dùng: Người lớn và trẻ em > 12 tuổi: 1 viên mỗi 6 giờ.
         ..."
    
    Args:
        product_name: Tên sản phẩm (ví dụ: "Thuốc Decolgen Forte United...")
        section_title: Tên phần (ví dụ: "Liều dùng")
        content: Nội dung text đã được clean HTML
    """
    # Lấy tên ngắn gọn hơn (bỏ phần trong ngoặc đơn cuối)
    # "Thuốc Decolgen Forte United giảm... (30 vỉ x 4 viên)" → "Thuốc Decolgen Forte United giảm..."
    short_name = product_name.split("(")[0].strip() if "(" in product_name else product_name
    
    return f"{short_name} — {section_title}:\n{content}"


# ============================================================
# HÀM TẠO CHUNK FAQ
# ============================================================
def build_faq_chunk(product_name: str, question: str, answer: str) -> str:
    """
    Tạo chunk cho 1 cặp hỏi-đáp (FAQ).
    
    TẠI SAO MỖI CÂU FAQ LÀ 1 CHUNK RIÊNG?
    Vì mỗi câu hỏi FAQ là 1 chủ đề độc lập. Ví dụ:
    - "Decolgen Forte có gây buồn ngủ không?" ← chủ đề: tác dụng phụ
    - "Phụ nữ mang thai có dùng được không?" ← chủ đề: đối tượng sử dụng
    
    Nếu gộp tất cả FAQ vào 1 chunk → embedding bị "loãng" vì quá nhiều chủ đề
    → khi user hỏi về buồn ngủ, chunk FAQ chung sẽ match kém hơn chunk FAQ riêng.
    
    Ví dụ output:
        "Thuốc Decolgen Forte — Hỏi đáp:
         Hỏi: Thuốc Decolgen Forte có gây buồn ngủ không?
         Đáp: Thuốc có thể gây kích thích thần kinh trung ương nhẹ, gây buồn ngủ..."
    """
    short_name = product_name.split("(")[0].strip() if "(" in product_name else product_name
    
    return f"{short_name} — Hỏi đáp:\nHỏi: {question}\nĐáp: {answer}"


# ============================================================
# HÀM CHÍNH: CHUNK 1 SẢN PHẨM
# ============================================================

# Định nghĩa mapping giữa field JSON → tên section tiếng Việt
# Thứ tự trong list này cũng là thứ tự tạo chunk
MEDICAL_SECTIONS = [
    # (tên field trong JSON,  tên section hiển thị)
    ("description",   "Mô tả"),
    ("usage",         "Công dụng"),
    ("dosage",        "Liều dùng"),
    ("side_effects",  "Tác dụng phụ"),
    ("precautions",   "Lưu ý và chống chỉ định"),
    ("storage",       "Bảo quản"),
]

def chunk_one_product(product: dict) -> list[dict]:
    """
    Chia 1 sản phẩm thành nhiều chunks.
    
    Cấu trúc output cho mỗi chunk:
    {
        "chunk_id":     "00049130_general_info",     ← ID duy nhất
        "product_sku":  "00049130",                  ← để truy ngược về sản phẩm
        "product_name": "Thuốc Decolgen Forte...",   ← để hiển thị
        "product_url":  "https://...",               ← để dẫn link cho user
        "section":      "general_info",              ← phần nào của sản phẩm
        "content":      "Thuốc Decolgen Forte..."    ← NỘI DUNG ĐỂ EMBEDDING
    }
    
    Trường "content" là trường DUY NHẤT sẽ được đưa vào embedding model.
    Các trường khác là metadata — dùng để hiển thị kết quả và lọc.
    
    Returns:
        Danh sách các chunk dict
    """
    chunks = []
    sku = product.get("sku", "unknown")
    name = product.get("name", "")
    url = product.get("url", "")
    
    # ── Chunk 1: Thông tin chung ──────────────────────────────
    # Gộp các trường ngắn (tên, giá, thành phần, danh mục...) thành 1 chunk
    general_content = build_general_info_chunk(product)
    if general_content.strip():
        chunks.append({
            "chunk_id": f"{sku}_general_info",
            "product_sku": sku,
            "product_name": name,
            "product_url": url,
            "section": "general_info",
            "content": general_content,
        })
    
    # ── Chunk 2-7: Các phần nội dung y tế ─────────────────────
    # Mỗi phần (công dụng, liều dùng, tác dụng phụ...) = 1 chunk riêng
    for field_name, section_title in MEDICAL_SECTIONS:
        content = product.get(field_name, "").strip()
        
        # BỎ QUA nếu nội dung trống — không tạo chunk rỗng
        # (một số sản phẩm thiếu mô tả hoặc tác dụng phụ)
        if not content:
            continue
        
        chunk_content = build_medical_content_chunk(name, section_title, content)
        chunks.append({
            "chunk_id": f"{sku}_{field_name}",
            "product_sku": sku,
            "product_name": name,
            "product_url": url,
            "section": field_name,
            "content": chunk_content,
        })
    
    # ── Chunk 8+: FAQ ─────────────────────────────────────────
    # Mỗi cặp hỏi-đáp = 1 chunk riêng (vì mỗi câu hỏi là 1 chủ đề khác nhau)
    faq_list = product.get("faq", [])
    for i, faq_item in enumerate(faq_list):
        question = faq_item.get("question", "").strip()
        answer = faq_item.get("answer", "").strip()
        
        if not question or not answer:
            continue
        
        chunk_content = build_faq_chunk(name, question, answer)
        chunks.append({
            "chunk_id": f"{sku}_faq_{i}",
            "product_sku": sku,
            "product_name": name,
            "product_url": url,
            "section": "faq",
            "content": chunk_content,
        })
    
    return chunks


# ============================================================
# HÀM CHẠY TOÀN BỘ
# ============================================================
def process_all_products():
    """
    Đọc tất cả file JSON đã xử lý → chunk → lưu thành 1 file chunks.json.
    
    TẠI SAO LƯU THÀNH 1 FILE DUY NHẤT?
    Vì bước tiếp theo (embedding) cần đọc tất cả chunks một lần
    để gửi batch vào embedding model. 1 file duy nhất dễ load hơn
    1019 file riêng lẻ.
    """
    # Tìm tất cả file JSON (bỏ qua chunks.json nếu đã tồn tại)
    json_files = [
        f for f in glob.glob(os.path.join(DATA_PROCESSED_DIR, "*.json"))
        if os.path.basename(f) != "chunks.json"
    ]
    
    print(f"Tìm thấy {len(json_files)} file sản phẩm.")
    
    all_chunks = []
    products_with_no_chunks = 0
    
    for i, filepath in enumerate(json_files, 1):
        with open(filepath, "r", encoding="utf-8") as f:
            product = json.load(f)
        
        chunks = chunk_one_product(product)
        
        if not chunks:
            products_with_no_chunks += 1
        
        all_chunks.extend(chunks)
        
        if i % 200 == 0 or i == len(json_files):
            print(f"  [{i}/{len(json_files)}] Đã chunk... (tổng chunks: {len(all_chunks)})")
    
    # Lưu tất cả chunks vào 1 file
    with open(CHUNKS_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    # ── Thống kê ──────────────────────────────────────────────
    # Đếm số chunk theo từng loại section
    section_counts = {}
    content_lengths = []
    for chunk in all_chunks:
        section = chunk["section"]
        section_counts[section] = section_counts.get(section, 0) + 1
        content_lengths.append(len(chunk["content"]))
    
    print(f"\n{'=' * 60}")
    print(f"HOÀN THÀNH!")
    print(f"{'=' * 60}")
    print(f"  Số sản phẩm:        {len(json_files)}")
    print(f"  Sản phẩm không chunk: {products_with_no_chunks}")
    print(f"  Tổng số chunks:     {len(all_chunks)}")
    print(f"  Trung bình chunks/SP: {len(all_chunks) / len(json_files):.1f}")
    print(f"\n  Phân bổ theo section:")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"    {section:30s} {count:5d} chunks")
    print(f"\n  Độ dài content:")
    print(f"    Ngắn nhất:  {min(content_lengths):,} ký tự")
    print(f"    Dài nhất:   {max(content_lengths):,} ký tự")
    print(f"    Trung bình: {sum(content_lengths) / len(content_lengths):,.0f} ký tự")
    print(f"\n  Output: {CHUNKS_OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    process_all_products()
