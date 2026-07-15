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

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Sistem Pencarian Dokumen Berita",
    page_icon="📰",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

.block-container{
    max-width:1100px;
    margin:auto;
    padding-top:2rem;
}

.stButton>button{
    width:100%;
    height:48px;
    font-size:18px;
    border-radius:10px;
}

.card{
    padding:20px;
    border-radius:12px;
    border:1px solid #DDDDDD;
    margin-bottom:20px;
}

</style>
""", unsafe_allow_html=True)

nltk.download("stopwords")

# =====================================================
# LOAD FILE
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(BASE_DIR, "Dataset-STR.csv")
MODEL = os.path.join(BASE_DIR, "ujimodelgr-300-9-0.01.model")
IMAGE = os.path.join(BASE_DIR, "berita3.jpg")


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET)


@st.cache_resource
def load_model():
    return Word2Vec.load(MODEL)


df = load_dataset()
model = load_model()

stop_words = set(stopwords.words("indonesian"))

# =====================================================
# PREPROCESSING
# =====================================================

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


# =====================================================
# QUERY EXPANSION
# =====================================================

def expand_query(query_tokens, topn=5):

    valid = [w for w in query_tokens if w in model.wv]

    if len(valid) == 0:
        return query_tokens, []

    if len(valid) == 1:
        similar = model.wv.most_similar(valid[0], topn=topn*2)

    else:
        vectors = [model.wv[w] for w in valid]
        avg = np.mean(vectors, axis=0)
        similar = model.wv.most_similar(avg, topn=topn*2)

    tambahan = []
    skor = []

    for w, s in similar:

        if w not in valid and w not in tambahan and w not in stop_words:
            tambahan.append(w)
            skor.append(s)

        if len(tambahan) >= topn:
            break

    return valid + tambahan, list(zip(tambahan, skor))


# =====================================================
# HEADER
# =====================================================

st.markdown(
"""
<h1 style='text-align:center;'>
📰 Sistem Pencarian Dokumen Berita
</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div style='text-align:center;font-size:18px;'>

Sistem pencarian dokumen berita menggunakan
<b>Skip-Gram Word2Vec</b>,
<b>Query Expansion</b>,
dan
<b>BM25+</b>
untuk meningkatkan relevansi hasil pencarian berita.

</div>
""",
unsafe_allow_html=True
)

st.write("")

if os.path.exists(IMAGE):

    col1,col2,col3 = st.columns([1,2,1])

    with col2:
        image = Image.open(IMAGE)
        st.image(image,use_container_width=True)

st.divider()

# =====================================================
# INPUT
# =====================================================

left,center,right = st.columns([1,3,1])

with center:

    query = st.text_input(
        "Masukkan Kata Kunci",
        placeholder="Contoh : pajak"
    )

    cari = st.button("🔍 Cari Berita")

# =====================================================
# SEARCH
# =====================================================

if cari:

    if query.strip()=="":

        st.warning("Masukkan query terlebih dahulu.")
        st.stop()

    tokens = preprocess_text(query)

    expanded_query, similar_words = expand_query(tokens)

    st.divider()

    st.subheader("🧩 Hasil Query Expansion")

    st.write("**Query Awal**")
    st.write(tokens)

    st.write("**Query Setelah Ekspansi**")
    st.write(expanded_query)

    if len(similar_words):

        st.table(
            pd.DataFrame(
                similar_words,
                columns=[
                    "Kata",
                    "Skor Similarity"
                ]
            )
        )

    corpus = df["berita"].astype(str).tolist()

    tokenized = [
        tokenizing(doc.lower())
        for doc in corpus
    ]

    bm25 = BM25Plus(tokenized)

    scores = bm25.get_scores(expanded_query)

    top = np.argsort(scores)[::-1][:10]

    st.divider()

    st.subheader("📰 Top 10 Hasil Pencarian")

    for i in top:

        row = df.iloc[i]

        with st.container(border=True):

            st.markdown(f"### {row['judul']}")

            c1,c2 = st.columns(2)

            with c1:
                st.write("📅",row["tanggal"])

            with c2:
                st.write("🏷️",row["kategori"])

            st.write(row["berita"][:400]+"...")

            st.success(f"Skor BM25+ : {scores[i]:.4f}")

            st.link_button(
                "🔗 Buka Berita",
                row["link"],
                use_container_width=True
            )
