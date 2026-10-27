import streamlit as st
from assistant.rag import RAGAssistant
from assistant.documents import load_uploaded_documents, split_documents
from assistant.config import DEFAULT_MODEL, EMBEDDING_MODEL

st.set_page_config(page_title="Personalized AI Assistant", page_icon="🤖", layout="wide")
st.title("Personalized AI Assistant")
st.caption("Open-source LLM • RAG • LoRA/QLoRA research starter")

with st.sidebar:
    model_name = st.text_input("Instruction model", DEFAULT_MODEL)
    embedding_name = st.text_input("Embedding model", EMBEDDING_MODEL)
    top_k = st.slider("Retrieved passages", 1, 8, 4)
    st.info("The first run downloads model weights. CPU inference may be slow.")

if ("assistant" not in st.session_state or st.session_state.get("model_name") != model_name
        or st.session_state.get("embedding_name") != embedding_name):
    st.session_state.assistant = RAGAssistant(model_name, embedding_name)
    st.session_state.model_name, st.session_state.embedding_name = model_name, embedding_name
if "messages" not in st.session_state:
    st.session_state.messages = []

chat_tab, docs_tab, research_tab = st.tabs(["Chat", "Your documents", "Research & evaluation"])
with docs_tab:
    st.write("Upload personal documents to retrieve relevant passages for your questions.")
    uploads = st.file_uploader("Upload PDF, DOCX, TXT, or Markdown", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("Index uploaded documents", type="primary", disabled=not uploads):
        with st.spinner("Reading and indexing..."):
            docs = load_uploaded_documents(uploads)
            chunks = split_documents(docs)
            st.session_state.assistant.index_documents(chunks)
            st.session_state.chunk_count = len(chunks)
        st.success(f"Indexed {len(chunks)} chunks.")
    st.write(f"Chunks indexed this session: {st.session_state.get('chunk_count', 0)}")

with chat_tab:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources used"):
                    for src in msg["sources"]:
                        st.markdown(f"- **{src['source']}** — {src['excerpt']}")
    question = st.chat_input("Ask about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating..."):
                answer, sources = st.session_state.assistant.answer(question, top_k)
            st.markdown(answer)
            if sources:
                with st.expander("Sources used"):
                    for src in sources:
                        st.markdown(f"- **{src['source']}** — {src['excerpt']}")
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

with research_tab:
    st.subheader("Project aim")
    st.write("Develop a personalized assistant with an open-source foundational model, RAG over user-specific documents, and an experiment measuring the effect of LoRA/QLoRA on response quality.")
    st.code("Question → embeddings/retrieval → relevant document context → open-source LLM → answer + source excerpts")
    st.write("See docs/EXPERIMENTS.md for the comparison conditions and evaluation rubric.")
