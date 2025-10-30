# Kelompok7_UTS_PraktikumPenelusuranInformasi
README - Sistem Information Retrieval UTS

Deskripsi
Sistem pencarian dokumen berbasis CLI yang membaca file CSV dari folder dataset/, melakukan preprocessing, indexing dengan Whoosh, dan ranking menggunakan Cosine Similarity.

Library yang Dipakai
- Whoosh (indexing & search)
- Pandas (baca CSV)
- Scikit-learn (vektorisasi & cosine similarity)
- Sastrawi (stemming bahasa Indonesia)

Install:
pip install pandas whoosh scikit-learn sastrawi


Struktur File
├── dataset/               # Taruh file CSV di sini
│   ├── etd_ugm.csv
│   ├── etd_usk.csv
│   ├── kompas.csv
│   ├── tempo.csv
│   └── mojok.csv
├── whoosh_index/          # Auto dibuat saat indexing
├── countvec.pkl           # Auto dibuat saat indexing
├── documents.pkl          # Auto dibuat saat indexing
└── main.py

Cara Jalankan

1. Install library
pip install pandas whoosh scikit-learn sastrawi

2. Jalankan program
python main.py

3. Menu yang muncul
=== SISTEM INFORMATION RETRIEVAL UTS ===
1. Indexing Dataset
2. Cari Query
3. Keluar

Panduan Penggunaan

PERTAMA KALI: Indexing Dulu (Menu 1)
1. Pilih 1
2. Program akan:
   - Baca semua CSV di folder dataset/
   - Preprocessing semua dokumen (lowercase, remove stopwords, stemming)
   - Buat index Whoosh
   - Simpan vectorizer ke file pkl
3. Output:
   [INFO] Membaca: dataset/kompas.csv
   [INFO] Total dokumen: 1500
   [INFO] Memulai preprocessing & vektorisasi...
   [Progress] Preprocessed 500/1500 dokumen...
   [INFO] Index Whoosh selesai dibuat.

Estimasi waktu:
- 100 dokumen: 10 detik
- 1000 dokumen: 1-2 menit
- 5000 dokumen: 5-10 menit

Pencarian (Menu 2)
1. Pilih 2
2. Masukkan query: `Kebijakan Pemerintah`
3. Output:
  
   === HASIL PENCARIAN ===
   [1] Judul Dokumen (score=0.8542)
        Snippet 200 karakter pertama...
   
   [2] Judul Dokumen (score=0.7891)
        Snippet...

Catatan: Setelah indexing pertama kali, langsung bisa pakai menu 2. TIDAK PERLU indexing lagi kecuali ada perubahan dataset.

## Cara Kerja Sistem

### Preprocessing
1. Lowercase semua teks
2. Hapus HTML tags
3. Tokenization (pisah per kata)
4. Hapus stopwords (yang, dan, di, ke, dari, the, and, is, dll)
5. Filter kata panjang > 2 karakter
6. Stemming pakai Sastrawi

### Indexing
1. Baca semua CSV dari folder dataset/
2. Deteksi kolom judul (judul/title/headline) dan isi (isi/content/text/abstrak)
3. Preprocessing tiap dokumen
4. Buat index Whoosh (simpan di whoosh_index/)
5. Vektorisasi dokumen pakai CountVectorizer (Bag of Words)
6. Simpan model ke countvec.pkl dan documents.pkl

### Pencarian & Ranking
1. User input query
2. Whoosh cari 50 dokumen kandidat
3. Preprocessing query (sama seperti dokumen)
4. Hitung Cosine Similarity antara query dan 50 kandidat
5. Ranking berdasarkan similarity score (tinggi ke rendah)
6. Tampilkan top-5 hasil

## Format CSV yang Didukung
CSV harus punya:
- Kolom judul: judul / title / headline
- Kolom isi: isi / content / text / body / abstrak / abstract

Program otomatis deteksi nama kolom.

## Fitur Optimasi
- Caching stemming: Kata yang sudah di-stem disimpan di cache, jadi gak perlu stem ulang
- Progress indicator: Tampil setiap 500 dokumen saat preprocessing
- Pickle storage: Model dan data disimpan, jadi gak perlu rebuild tiap jalankan program

Score berkisar 0-1, makin tinggi makin relevan dengan query.
