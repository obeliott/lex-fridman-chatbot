import os
# fall back to the pure-python protobuf parser so chromadb's opentelemetry
# imports don't choke on the older _pb2 files (cloud uses protobuf 7+)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from pathlib import Path

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


CHROMA_DIR = Path("chroma_db")
COLLECTION = "lex_fridman"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant that answers questions about the Lex Fridman Podcast.
Use only the transcript excerpts below to answer. If the excerpts don't contain
the answer, say you don't know rather than guessing. Quote or paraphrase the
guests directly when relevant.

Excerpts:
{context}

Question: {question}

Answer:"""
)


def get_llm():
    # GROQ_API_KEY in env -> use Groq (cloud-friendly), otherwise fall back
    # to a local Ollama model. Streamlit Cloud uses the Groq path.
    if os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None):
        from langchain_groq import ChatGroq
        # Streamlit Cloud injects secrets into env, but be safe locally
        if not os.getenv("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.2,
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "mistral"),
        temperature=0.2,
    )


@st.cache_resource(show_spinner="loading podcast index...")
def get_retriever():
    embed = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=embed,
        persist_directory=str(CHROMA_DIR),
    )
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 16})


def format_context(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        ep = d.metadata.get("episode", "?")
        parts.append(f"[Excerpt {i} - {ep}]\n{d.page_content}")
    return "\n\n".join(parts)


def pretty_timestamp(url):
    # source urls look like https://karpathy.ai/lexicap/0045-large.html#00:58:47.360
    # the karpathy site is dead now, but the fragment still tells us where in
    # the episode we are
    import urllib.parse as u
    frag = u.urlparse(url).fragment
    if not frag:
        return None
    # trim milliseconds and leading zero hour
    parts = frag.split('.')[0].split(':')
    if len(parts) == 3 and parts[0] == '00':
        return ':'.join(parts[1:])
    return ':'.join(parts)


def search_link(episode, timestamp):
    # the karpathy lexicap pages are gone, so just point at a google search
    # for the episode + timestamp (lands on the episode's youtube video or
    # the official transcript on lexfridman.com)
    import urllib.parse as u
    q = f'"Lex Fridman Podcast" {episode}'
    if timestamp:
        q += f' {timestamp}'
    return 'https://www.google.com/search?q=' + u.quote_plus(q)


def answer(question, retriever, llm):
    docs = retriever.invoke(question)
    ctx = format_context(docs)
    msg = PROMPT.format(context=ctx, question=question)
    resp = llm.invoke(msg)
    text = resp.content if hasattr(resp, "content") else str(resp)
    return text, docs


# ---- Streamlit UI ----

st.set_page_config(page_title="Lex Fridman Podcast Chatbot", page_icon=":microphone:")
st.title("Lex Fridman Podcast Chatbot")
st.caption("Ask questions about anything discussed on the Lex Fridman Podcast.")

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "RAG chatbot built on transcripts of the Lex Fridman Podcast. "
        "Embeddings are generated with `all-MiniLM-L6-v2` and stored in Chroma. "
        "The LLM is Llama 3 via Groq when deployed, or Mistral via Ollama locally."
    )
    st.markdown("### Try asking")
    st.markdown(
        "- What did Elon Musk say about Mars?\n"
        "- Has Lex talked to anyone about consciousness?\n"
        "- What is Yann LeCun's view on LLMs?\n"
        "- Did anyone discuss the future of AGI?"
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

retriever = get_retriever()
llm = get_llm()

def render_sources(srcs):
    for s in srcs:
        ts = s.get("timestamp")
        link = s.get("search_url", "")
        ts_str = f" @ {ts}" if ts else ""
        st.markdown(f"- **{s['episode']}**{ts_str} - [find episode]({link})")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                render_sources(m["sources"])

q = st.chat_input("Ask something about the podcast...")
if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            text, docs = answer(q, retriever, llm)
        st.markdown(text)
        srcs = []
        for d in docs:
            ep = d.metadata.get("episode", "?")
            ts = pretty_timestamp(d.metadata.get("source_url", ""))
            srcs.append({
                "episode": ep,
                "timestamp": ts,
                "search_url": search_link(ep, ts),
            })
        with st.expander("Sources"):
            render_sources(srcs)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": srcs}
    )
