# Retrieval Integrity Auditor

This project audits the retrieval step of Retrieval-Augmented Generation (RAG) pipelines by evaluating coverage, identifying missing evidence, and detecting retrieval noise before answer generation.

## Key Features
- Semantic coverage analysis of retrieved chunks
- Explicit identification of missing query aspects
- Retrieval noise detection
- Explainable audit results with visualizations
- Model-agnostic and domain-independent design

## How It Works
1. The user query is analyzed to determine expected information aspects.
2. Retrieved chunks are matched to these aspects using semantic similarity.
3. Coverage gaps and irrelevant chunks are identified before generation.

## Tech Stack
- Python
- Streamlit
- Sentence Transformers

## Note
This tool focuses on retrieval auditing and does not perform answer generation.

