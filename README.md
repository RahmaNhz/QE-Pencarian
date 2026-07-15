# 📰 Sistem Pencarian Dokumen Berita Menggunakan Query Expansion Berbasis Skip-Gram Word2Vec dan BM25+

Aplikasi ini merupakan implementasi sistem Information Retrieval (IR) yang dikembangkan sebagai bagian dari penelitian skripsi.

Sistem menggunakan metode **Query Expansion berbasis Skip-Gram Word2Vec** untuk memperluas kata kunci pencarian, kemudian melakukan pemeringkatan dokumen menggunakan **BM25+** sehingga hasil pencarian menjadi lebih relevan.

## 🚀 Demo

Aplikasi dapat dijalankan melalui Streamlit Community Cloud. "https://qe-pencarian-berita.streamlit.app/"

## 📌 Fitur

- Pencarian dokumen berita
- Preprocessing query
  - Case Folding
  - Tokenizing
  - Stopword Removal
- Query Expansion menggunakan Skip-Gram Word2Vec
- Pemeringkatan dokumen menggunakan BM25+
- Menampilkan 10 dokumen berita paling relevan
- Menampilkan skor BM25+ setiap dokumen

---

## 🛠️ Teknologi

- Python
- Streamlit
- Gensim
- Rank-BM25
- NLTK
- Pandas
- NumPy

---

## 📂 Struktur Project

```
.
├── app.py
├── Dataset-STR.csv
├── ujimodelgr-300-9-0.01.model
├── berita3.jpg
├── requirements.txt
└── README.md
```

---

## ⚙️ Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/nama-repository.git
```

### 2. Masuk Folder

```bash
cd nama-repository
```

### 3. Install Library

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

---

## 📄 Dataset

Dataset terdiri dari kumpulan berita Indonesia yang telah melalui tahap preprocessing dan digunakan sebagai korpus pada proses Information Retrieval.

---

## 🧠 Model

Model Word2Vec yang digunakan:

- Arsitektur : Skip-Gram
- Vector Size : 300
- Window Size : 9
- Learning Rate : 0.01

---

## 🔍 Metode

1. Input Query
2. Case Folding
3. Tokenizing
4. Stopword Removal
5. Query Expansion (Skip-Gram Word2Vec)
6. BM25+
7. Ranking Dokumen
8. Menampilkan Top 10 Berita

---

## 👩‍🎓 Penelitian

Aplikasi ini dikembangkan sebagai implementasi penelitian dengan judul:

**"Pengaruh Query Expansion Berbasis Skip-Gram Word2Vec untuk Meningkatkan Relevansi Pencarian Dokumen Berita Menggunakan BM25+"**

---

## 📜 Lisensi

Project ini dibuat untuk keperluan penelitian akademik.
