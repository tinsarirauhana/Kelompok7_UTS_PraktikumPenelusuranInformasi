#!/usr/bin/env python3
"""
UTS Praktikum PI - CLI Information Retrieval System
- Membaca folder dataset/ berisi file CSV (etd_ugm.csv, etd_usk.csv, kompas.csv, tempo.csv, mojok.csv)
- Preprocessing, Indexing (Whoosh), Search + Ranking (Cosine Similarity)
"""

import os, re, sys, pickle
import pandas as pd
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StandardAnalyzer
from whoosh.qparser import QueryParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Tambahan untuk Stemming ----------
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ---------- Konfigurasi ----------
DATASET_DIR = "dataset"
INDEX_DIR = "whoosh_index"
VEC_STORE = "countvec.pkl"
DOCS_STORE = "documents.pkl"

STOPWORDS = {
    "yang","dan","di","ke","dari","ini","itu","pada","untuk","dengan","adalah","sebagai",
    "oleh","karena","dalam","akan","atau","juga","tidak","lebih","dapat","dengan","para",
    "the","and","is","in","of","to","a","for","on","that","this","as","by","from","be"
}

# ---------- Preprocessing ----------
# Tambahkan cache untuk mempercepat stemming
_stem_cache = {}

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    tokens = re.findall(r"\b[\w']+\b", text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    processed_tokens = []
    for t in tokens:
        if t in _stem_cache:
            stemmed = _stem_cache[t]
        else:
            stemmed = stemmer.stem(t)
            _stem_cache[t] = stemmed
        processed_tokens.append(stemmed)

    return " ".join(processed_tokens)


# ---------- Baca CSV Dataset ----------
def load_csv_documents(folder=DATASET_DIR):
    docs = []
    if not os.path.exists(folder):
        print(f"[ERROR] Folder '{folder}' tidak ditemukan.")
        return docs

    doc_id = 0
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            path = os.path.join(folder, file)
            print(f"[INFO] Membaca: {path}")
            try:
                df = pd.read_csv(path, dtype=str, encoding="utf-8")
            except Exception:
                df = pd.read_csv(path, dtype=str, encoding="latin1")
            df = df.fillna("")

            # cari kolom yang mirip 'judul', 'title', 'headline'
            title_col = next((c for c in df.columns if c.lower() in 
                              ["judul","title","headline"]), None)
            # cari kolom isi
            content_col = next((c for c in df.columns if c.lower() in 
                                ["isi","content","text","body","abstrak","abstract"]), None)
            
            for _, row in df.iterrows():
                title = str(row[title_col]) if title_col else f"{file}_{doc_id}"
                content = str(row[content_col]) if content_col else " ".join([str(x) for x in row.values])
                docs.append({
                    "doc_id": str(doc_id),
                    "title": title,
                    "content": content,
                    "path": path
                })
                doc_id += 1

    print(f"[INFO] Total dokumen: {len(docs)}")
    return docs

# ---------- Whoosh Index ----------
def create_index(docs):
    if not os.path.exists(INDEX_DIR):
        os.mkdir(INDEX_DIR)
    schema = Schema(
        doc_id=ID(stored=True, unique=True),
        title=TEXT(stored=True),
        path=ID(stored=True),
        content=TEXT(stored=True, analyzer=StandardAnalyzer())
    )
    ix = index.create_in(INDEX_DIR, schema)
    writer = ix.writer()
    for d in docs:
        writer.add_document(
            doc_id=d["doc_id"],
            title=d["title"],
            path=d["path"],
            content=d["content"]
        )
    writer.commit()
    print("[INFO] Index Whoosh selesai dibuat.")
    return ix

# ---------- Vektor Dokumen ----------
def build_vectorizer(docs):
    contents = []
    total_docs = len(docs)
    print(f"[INFO] Memulai preprocessing & vektorisasi ({total_docs} dokumen)...")

    for i, d in enumerate(docs, start=1):
        contents.append(preprocess(d["content"]))
        if i % 500 == 0 or i == total_docs:
            print(f"   [Progress] Preprocessed {i}/{total_docs} dokumen...")

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(contents)
    doc_map = {d["doc_id"]: d for d in docs}

    with open(VEC_STORE, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "X": X}, f)
    with open(DOCS_STORE, "wb") as f:
        pickle.dump(doc_map, f)

    print("[INFO] Representasi BoW disimpan.")
    return vectorizer, X, doc_map

# ---------- Search + Ranking ----------
def search_and_rank(ix, query, vectorizer, X, doc_map, top_k=5):
    qp = QueryParser("content", ix.schema)
    q = qp.parse(query)
    with ix.searcher() as s:
        results = s.search(q, limit=50)
        if not results:
            print("Tidak ada hasil dari Whoosh.")
            return []

        candidates = [r["doc_id"] for r in results]
        q_vec = vectorizer.transform([preprocess(query)])
        doc_texts = [preprocess(doc_map[d]["content"]) for d in candidates]
        mat = vectorizer.transform(doc_texts)
        sims = cosine_similarity(q_vec, mat)[0]
        ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

# ---------- Tampilkan Hasil ----------
def show_results(results, doc_map):
    print("\n=== HASIL PENCARIAN ===")
    for i, (doc_id, score) in enumerate(results, start=1):
        doc = doc_map[doc_id]
        print(f"[{i}] {doc['title']} (score={score:.4f})")
        snippet = doc["content"].replace("\n", " ")[:200]
        print(f"     {snippet}...\n")

# ---------- CLI ----------
def main():
    ix, vectorizer, X, doc_map = None, None, None, None
    while True:
        print("\n=== SISTEM INFORMATION RETRIEVAL UTS ===")
        print("1. Indexing Dataset")
        print("2. Cari Query")
        print("3. Keluar")
        choice = input("Pilih (1/2/3): ").strip()

        if choice == "1":
            docs = load_csv_documents(DATASET_DIR)
            if not docs:
                continue
            ix = create_index(docs)
            vectorizer, X, doc_map = build_vectorizer(docs)

        elif choice == "2":
            if ix is None or vectorizer is None:
                if index.exists_in(INDEX_DIR):
                    ix = index.open_dir(INDEX_DIR)
                vectorizer, X, doc_map = load_vectorizer()
            if ix is None:
                print("[!] Belum ada index. Jalankan menu 1 dulu.")
                continue
            query = input("Masukkan query: ").strip()
            if not query:
                continue
            results = search_and_rank(ix, query, vectorizer, X, doc_map)
            show_results(results, doc_map)

        elif choice == "3":
            print("Selesai.")
            break
        else:
            print("Pilihan tidak valid.")

if __name__ == "__main__":
    main()
