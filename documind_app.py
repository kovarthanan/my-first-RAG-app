import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # optional: LangSmith keys from .env

# ---------- Settings ----------
APP_NAME = "DocuMind"
PDF_FOLDER = "pdfs"
INDEX_FOLDER = "faiss_index"
LLM_MODEL = "qwen2.5:1.5b"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 50

st.set_page_config(page_title=APP_NAME, page_icon="✨", layout="wide")

# ---------- Look ----------
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {max-width: 850px; padding-top: 2rem;}

    .gradient-title {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 3rem; font-weight: 500; line-height: 1.2;
        color: #C96442;
    }
    .sub-title {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.8rem; font-weight: 400; color: #83827D;
        margin-bottom: 1.5rem;
    }

    [data-testid="stChatInput"] {
        border-radius: 20px; border: 1px solid #E3DFD3;
        box-shadow: 0 2px 8px rgba(61, 57, 41, 0.06);
    }
    [data-testid="stChatInput"] textarea {font-size: 1rem;}
    [data-testid="stChatMessage"] {background: transparent;}

    section[data-testid="stSidebar"] {border-right: 1px solid #E3DFD3;}
    .stButton > button {
        border-radius: 12px; border: 1px solid #D9D5C7;
        background: #FFFFFF; width: 100%;
    }
    .stButton > button:hover {border-color: #C96442; color: #C96442;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Models (loaded once) ----------
@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)


@st.cache_resource
def get_chain():
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on the documents."
Keep the answer short and clear.

Context:
{context}

Question: {question}

Answer:"""
    )
    return prompt | llm | StrOutputParser()


# ---------- Index ----------
def load_index():
    if Path(INDEX_FOLDER).exists():
        return FAISS.load_local(
            INDEX_FOLDER, get_embeddings(), allow_dangerous_deserialization=True
        )
    return None


def build_index():
    pdf_files = sorted(Path(PDF_FOLDER).glob("*.pdf"))
    if not pdf_files:
        st.sidebar.error("No PDFs found. Upload some first.")
        return None

    docs = []
    for pdf in pdf_files:
        docs.extend(PyPDFLoader(str(pdf)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        st.sidebar.error("No text found. Your PDFs may be scanned images.")
        return None

    bar = st.sidebar.progress(0, text="Starting...")
    store = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        if store is None:
            store = FAISS.from_documents(batch, get_embeddings())
        else:
            store.add_documents(batch)
        done = min(i + BATCH_SIZE, len(chunks))
        bar.progress(done / len(chunks), text=f"Reading {done}/{len(chunks)} chunks")

    store.save_local(INDEX_FOLDER)
    bar.empty()
    return store


# ---------- Session ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = load_index()

Path(PDF_FOLDER).mkdir(exist_ok=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(f"### ✨ {APP_NAME}")
    st.caption("Chat with your PDFs. Runs fully on your laptop.")

    uploaded = st.file_uploader(
        "Add PDFs", type="pdf", accept_multiple_files=True
    )

    if st.button("Update index"):
        for f in uploaded or []:
            (Path(PDF_FOLDER) / f.name).write_bytes(f.getbuffer())
        new_store = build_index()
        if new_store:
            st.session_state.store = new_store
            st.toast("Index updated", icon="✅")

    top_k = st.slider("Chunks to use", 1, 6, 3)

    pdfs = sorted(p.name for p in Path(PDF_FOLDER).glob("*.pdf"))
    if pdfs:
        st.markdown("**Documents**")
        for name in pdfs:
            st.caption(f"📄 {name}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Main ----------
if not st.session_state.messages:
    st.markdown('<div class="gradient-title">Hello there</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">What would you like to know from your documents?</div>',
        unsafe_allow_html=True,
    )

if st.session_state.store is None:
    st.info("Add PDFs in the sidebar and click **Update index** to start.")
    st.stop()

# show chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="✨" if m["role"] == "assistant" else "🧑"):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.caption(f"📄 {s}")

# new question
if question := st.chat_input(f"Ask {APP_NAME}"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="✨"):
        start = time.time()
        status = st.status("Searching your documents...", expanded=False)

        docs = st.session_state.store.similarity_search(question, k=top_k)
        context = "\n\n".join(d.page_content for d in docs)
        status.update(label=f"Found {len(docs)} relevant parts. Thinking...")

        def stream_answer():
            first = True
            for token in get_chain().stream({"context": context, "question": question}):
                if first:
                    status.update(label="Writing answer...")
                    first = False
                yield token
            status.update(
                label=f"Done in {time.time() - start:.0f}s",
                state="complete",
                expanded=False,
            )

        answer = st.write_stream(stream_answer())

        sources = sorted(
            {
                f"{Path(d.metadata.get('source', 'unknown')).name}, page {d.metadata.get('page', 0) + 1}"
                for d in docs
            }
        )
        with st.expander("Sources"):
            for s in sources:
                st.caption(f"📄 {s}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
