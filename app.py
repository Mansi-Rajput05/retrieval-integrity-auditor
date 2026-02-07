import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
from sentence_transformers import SentenceTransformer, util

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Retrieval Integrity Auditor", layout="wide")

st.title("🔍 Retrieval Integrity Auditor for RAG Systems")
st.write(
    "This system audits the retrieval step of a RAG pipeline by evaluating coverage, identifying missing evidence, and detecting retrieval noise before generation."
)


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------------------------------
# ASPECT SCHEMA
# -------------------------------------------------
ASPECT_DESCRIPTIONS = {
    "Dosage": "recommended amount and frequency at which a drug should be taken",
    "Side Effects": "unintended or adverse effects caused by a drug",
    "Drug Interactions": "how a drug interacts with or interferes with other medications",
    "Contraindications": "conditions where a drug should not be used"
}

INTERACTION_TRIGGERS = [
    "interact", "interaction", "interferes", "co-administer", "combine with"
]

STRONG_THRESHOLD = 0.55
WEAK_THRESHOLD = 0.45
RETRIEVAL_THRESHOLD = 0.70

# -------------------------------------------------
# USER INPUT
# -------------------------------------------------
st.header("1️⃣ User Input")

query = st.text_input(
    "Enter User Query",
    value="What are the dosage, side effects, contraindications, and drug interactions of Drug X?"
)

default_chunks = [
    {
        "id": "C1",
        "text": "Drug X should be taken at a dosage of 10mg twice daily for adults.",
        "similarity_score": 0.84
    },
    {
        "id": "C2",
        "text": "Common side effects of Drug X include nausea, headache, and dizziness.",
        "similarity_score": 0.81
    },
    {
        "id": "C3",
        "text": "Drug X may interact with anticoagulant medications, increasing the risk of bleeding.",
        "similarity_score": 0.77
    },
    {
        "id": "C4",
        "text": "Drug X was approved by regulatory authorities in 1998.",
        "similarity_score": 0.42
    }
]

chunks_input = st.text_area(
    "Retrieved Chunks (JSON)",
    value=json.dumps(default_chunks, indent=2),
    height=260
)

retrieved_chunks = json.loads(chunks_input)

# -------------------------------------------------
# ASPECT DETECTION FROM QUERY
# -------------------------------------------------
def detect_aspects(query):
    q_emb = model.encode(query, convert_to_tensor=True)
    aspects = []

    for aspect, desc in ASPECT_DESCRIPTIONS.items():
        a_emb = model.encode(desc, convert_to_tensor=True)
        if util.cos_sim(q_emb, a_emb).item() >= 0.4:
            aspects.append(aspect)

    return aspects

# -------------------------------------------------
# COVERAGE ANALYSIS (FINAL FIX)
# -------------------------------------------------
def analyze_coverage(aspects, chunks):
    coverage = {a: [] for a in aspects}
    noise = []

    for chunk in chunks:
        text = chunk["text"].lower()
        text_emb = model.encode(chunk["text"], convert_to_tensor=True)

        # 🔥 SEMANTIC ROLE GATING
        if any(trigger in text for trigger in INTERACTION_TRIGGERS):
            if "Drug Interactions" in aspects:
                coverage["Drug Interactions"].append(chunk["id"])
                continue

        # Otherwise fallback to semantic similarity
        best_aspect = None
        best_score = 0

        for aspect in aspects:
            aspect_emb = model.encode(
                ASPECT_DESCRIPTIONS[aspect], convert_to_tensor=True
            )
            sim = util.cos_sim(text_emb, aspect_emb).item()

            if sim > best_score:
                best_score = sim
                best_aspect = aspect

        if (
            best_score >= STRONG_THRESHOLD
            or (
                best_score >= WEAK_THRESHOLD
                and chunk["similarity_score"] >= RETRIEVAL_THRESHOLD
            )
        ):
            coverage[best_aspect].append(chunk["id"])
        else:
            noise.append(chunk["id"])

    return coverage, noise

# -------------------------------------------------
# SCORE
# -------------------------------------------------
def integrity_score(coverage):
    total = len(coverage)
    covered = sum(1 for a in coverage if coverage[a])
    return round((covered / total) * 100, 2)

# -------------------------------------------------
# RUN
# -------------------------------------------------
aspects = detect_aspects(query)
coverage, noise_chunks = analyze_coverage(aspects, retrieved_chunks)
missing = [a for a in coverage if not coverage[a]]
score = integrity_score(coverage)

# -------------------------------------------------
# RESULTS
# -------------------------------------------------
st.header("2️⃣ Audit Results")

st.metric("Retrieval Integrity Score", f"{score}/100")

if missing:
    st.error(f"Missing Aspects: {', '.join(missing)}")
else:
    st.success("All aspects covered")

# -------------------------------------------------
# HEATMAP
# -------------------------------------------------
st.header("3️⃣ Coverage Heatmap")

df = pd.DataFrame(
    [[1 if c["id"] in coverage[a] else 0 for c in retrieved_chunks] for a in aspects],
    index=aspects,
    columns=[c["id"] for c in retrieved_chunks]
)

fig, ax = plt.subplots()
ax.imshow(df, cmap="Greens")
ax.set_xticks(range(len(df.columns)))
ax.set_xticklabels(df.columns)
ax.set_yticks(range(len(df.index)))
ax.set_yticklabels(df.index)

for i in range(len(df.index)):
    for j in range(len(df.columns)):
        ax.text(j, i, df.iloc[i, j], ha="center", va="center")

st.pyplot(fig)

# -------------------------------------------------
# JSON OUTPUT
# -------------------------------------------------
st.header("4️⃣ Audit JSON")

st.json({
    "query": query,
    "coverage": coverage,
    "missing_aspects": missing,
    "noise_chunks": noise_chunks,
    "score": score
})

# -------------------------------------------------
# EXPLANATION
# -------------------------------------------------
st.header("5️⃣ Human-Readable Explanation")

st.write(f"""
- Each retrieved chunk is assigned to **only its best matching aspect**.
- This prevents semantic overlap from masking missing evidence.
- {len(missing)} critical aspect(s) lack supporting evidence.
- {len(noise_chunks)} chunk(s) were flagged as retrieval noise.
- Final retrieval integrity score: **{score}/100**.
""")
