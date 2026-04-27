# Lex Fridman Podcast Chatbot

A small RAG chatbot that answers questions about the Lex Fridman Podcast.

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store:** Chroma (persisted in `chroma_db/`)
- **LLM:** Llama 3 via Groq (cloud) or Mistral via Ollama (local)
- **UI:** Streamlit

## Quick start

```bash
pip install -r requirements.txt

# 1. Build the index (one-off, takes a few minutes on CPU)
python ingest.py            # all 319 episodes
# or
python ingest.py 50         # first 50 episodes for a smaller index

# 2. Run the app
#    Local with Ollama:
ollama pull mistral
streamlit run app.py

#    Or with Groq:
export GROQ_API_KEY=...
streamlit run app.py
```

## Files

- `ingest.py` builds the Chroma index from `data/lex_fridman_podcast.parquet`
- `app.py` is the Streamlit chatbot
- `chroma_db/` is the persisted vector store (committed so the cloud app
  doesn't have to re-embed at boot)

## Data

`data/lex_fridman_podcast.parquet` is the
[lambdaofgod/lex_fridman_podcast](https://huggingface.co/datasets/lambdaofgod/lex_fridman_podcast)
HF dataset (one row per sentence with timestamp links back to
karpathy.ai/lexicap).
