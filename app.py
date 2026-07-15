import streamlit as st
import pandas as pd
import numpy as np
import re
import os

from PIL import Image
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from rank_bm25 import BM25Plus
import nltk

# ======================
# CONFIG
# ======================

st.set_page_config(
    page_title="Pencarian Berita",
    page_icon="📰",
    layout="wide"
)

nltk.download("stopwords")

# ======================
# LOAD FILE
# ======================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(BASE_DIR, "Dataset-STR.csv")
MODEL = os.path.join(BASE_DIR, "ujimodelgr-300-9-0.01.model")
IMAGE = os.path.join(BASE_DIR, "berita3.jpg")

df = pd.read_csv(DATASET)

stop_words = set(stopwords.words("indonesian"))

model = Word2Vec.load(MODEL)

# ======================
# PREPROCESSING
# ======================

def case_folding(text):
    return text.lower()


def tokenizing(text):
    return re.findall(r"\b\w+\b", text)


def stopword_removal(tokens):
    return [t for t in tokens if t not in stop_words]


def preprocess_text(text):
    text = case_folding(text)
    tokens = tokenizing(text)
    tokens = stopword_removal(tokens)
    return tokens


# ======================
# QUERY EXPANSION
# ======================

def expand_query(query_tokens, topn=5):

    valid = [w for w in query_tokens if w in model.wv]

    if len(valid) == 0:
        return query_tokens, []

    if len(valid) == 1:
        similar = model.wv.most_similar(valid[0], topn=topn * 2)
    else:
        vectors = [model.wv[w] for w in valid]
        avg = np.mean(vectors, axis=0)
        similar = model.wv.most_similar(avg, topn=topn * 2)

    tambahan = []
    skor = []

    for w, s in similar:

        if (
            w not in valid
            and w not in stop_words
            and w not in tambahan
        ):
            tambahan.append(w)
            skor.append(s)

        if len(tambahan) >= topn:
            break

    return valid + tambahan, list(zip(tambahan, skor))


# ======================
# HEADER
# ======================

st.title("📰 Sistem Pencarian Dokumen Berita")

st.write(
"""
Sistem menggunakan

- Skip-Gram Word2Vec
- Query Expansion
- BM25+

untuk meningkatkan relevansi pencarian berita.
"""
)

if os.path.exists(IMAGE):
    image = Image.open(IMAGE)
    st.image(image, width=600)

st.divider()

# ======================
# INPUT QUERY
# ======================

query = st.text_input(
    "Masukkan kata kunci pencarian",
    placeholder="contoh: pajak"
)

if st.button("🔍 Cari Berita"):

    if query.strip() == "":
        st.warning("Masukkan query terlebih dahulu.")
        st.stop()

    tokens = preprocess_text(query)

    expanded_query, similar_words = expand_query(tokens)

    st.subheader("Hasil Query Expansion")

    st.write("Query awal :", tokens)

    st.write("Query setelah ekspansi :", expanded_query)

    if len(similar_words):

        st.table(
            pd.DataFrame(
                similar_words,
                columns=["Kata", "Skor Similarity"]
            )
        )

    st.divider()

    corpus = df["berita"].astype(str).tolist()

    tokenized = [
        tokenizing(doc.lower())
        for doc in corpus
    ]

    bm25 = BM25Plus(tokenized)

    scores = bm25.get_scores(expanded_query)

    top = np.argsort(scores)[::-1][:10]

    st.subheader("Top 10 Berita")

    for i in top:

        row = df.iloc[i]

        st.markdown(f"### {row['judul']}")

        c1, c2 = st.columns(2)

        c1.write(f"📅 {row['tanggal']}")

        c2.write(f"🏷️ {row['kategori']}")

        st.write(row["berita"][:400] + "...")

        st.write(f"**Skor BM25+ : {scores[i]:.4f}**")

        st.link_button(
            "Buka Berita",
            row["link"]
        )

        st.divider()
