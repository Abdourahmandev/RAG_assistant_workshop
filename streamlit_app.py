"""Local Streamlit interface: prepare once, then chat with cited document sources."""

from pathlib import Path

import streamlit as st

from rag.application import PROJECT_DIR, inspect_pdf, prepare_assistant, safe_error


@st.cache_data(show_spinner=False)
def preview_pdf(path: str, modified_ns: int, byte_count: int, size: int, overlap: int):
    # File metadata invalidates the local preview when a PDF is replaced.
    return inspect_pdf(Path(path), size, overlap)


def show_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("passages"):
            with st.expander("View retrieved passages"):
                for passage in message["passages"]:
                    st.markdown(f"**{passage['title']}** · `{passage['id']}`")
                    st.write(passage["text"])
            st.caption(f"Answered in {message['total_s']:.1f} s")


def release_assistant():
    assistant = st.session_state.get("assistant")
    if assistant is not None:
        assistant.close()
    st.session_state.assistant = None
    st.session_state.messages = []


def request_preparation():
    # Consume the rebuild option once. A retry must resume the new generation.
    st.session_state.requested_force_rebuild = st.session_state.get("force_rebuild", False)
    st.session_state.force_rebuild = False


def main():
    st.set_page_config(page_title="Document assistant", page_icon="📚", layout="wide")
    st.title("Document assistant")
    st.caption("Prepare your PDF, then ask questions. Answers use evidence from the document.")
    st.session_state.setdefault("assistant", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("active_signature", None)
    pdfs = sorted(PROJECT_DIR.glob("*.pdf"))
    if not pdfs:
        release_assistant()
        st.info("Add a PDF to the project folder, then reload this page.")
        st.stop()

    with st.sidebar:
        st.header("Your document")
        selected = st.selectbox("Project PDF", [path.name for path in pdfs], key="pdf")
        with st.expander("Settings"):
            size = st.number_input("Words per passage", min_value=20, max_value=300, value=80, step=10)
            overlap = st.number_input("Overlap in words", min_value=0, max_value=299, value=15)
            k = st.number_input("Passages per answer", min_value=1, max_value=8, value=3)
            st.checkbox("Recompute embeddings", value=False, key="force_rebuild",
                                help="Use this if the embedding model was redeployed or you want to recompute the vectors.")
        st.caption("Models use the configuration in the project .env file.")

    try:
        path = PROJECT_DIR / selected
        stat = path.stat()
        plan = preview_pdf(str(path), stat.st_mtime_ns, stat.st_size, int(size), int(overlap))
    except Exception as exc:
        release_assistant()
        st.error("Could not read the PDF: " + safe_error(exc))
        st.stop()

    if st.session_state.active_signature != plan.signature:
        release_assistant()
        st.session_state.active_signature = plan.signature

    columns = st.columns(3)
    columns[0].metric("Pages with text", len(plan.documents))
    columns[1].metric("Passages", len(plan.chunks))
    state_label = "Ready" if st.session_state.assistant else "Not prepared"
    state_slot = columns[2].empty()
    state_slot.metric("Assistant", state_label)

    prepare_clicked = st.button("Start processing", type="primary", key="prepare",
                                on_click=request_preparation)
    st.caption("Initial preparation may take a few minutes. Completed batches are saved and reused on later runs.")
    if prepare_clicked:
        force = st.session_state.pop("requested_force_rebuild", False)
        release_assistant()
        state_slot.metric("Assistant", "Preparing...")
        with st.status("Preparing the document", expanded=True) as status:
            message_slot = st.empty()
            progress_bar = st.progress(0.0)

            def on_progress(event):
                message_slot.write(event.message)
                if event.total:
                    progress_bar.progress(event.completed / event.total)

            try:
                assistant = prepare_assistant(
                    plan, force_rebuild=force, progress=on_progress,
                    database_dir=PROJECT_DIR / "base_vectorielle" / "streamlit",
                )
            except Exception as exc:
                status.update(label="Preparation interrupted", state="error")
                state_slot.metric("Assistant", "Retry needed")
                st.error(safe_error(exc))
                st.info("Completed batches are saved. Start processing again to resume.")
            else:
                st.session_state.assistant = assistant
                state_slot.metric("Assistant", "Ready")
                status.update(label="Document ready - you can chat now", state="complete", expanded=False)

    assistant = st.session_state.assistant
    if assistant:
        st.success(f"{len(plan.chunks)} passages ready · {assistant.cached_passages} reused from cache.")
    else:
        st.info("Click Start processing to enable chat.")
    if st.button("New conversation", disabled=not st.session_state.messages, key="clear_chat"):
        st.session_state.messages = []

    for message in st.session_state.messages:
        show_message(message)

    question = st.chat_input("Ask a question about your PDF...", disabled=assistant is None,
                             max_chars=2000, key="question", submit_mode="disable")
    if question and assistant:
        history = [{"role": msg["role"], "content": msg["content"]}
                   for msg in st.session_state.messages[-6:]]
        show_message({"role": "user", "content": question})
        with st.spinner("Retrieving passages and writing the answer..."):
            try:
                result = assistant.answer(question, history, k=int(k))
            except Exception as exc:
                st.error("Could not generate the answer: " + safe_error(exc))
                st.caption("The document is still ready. You can retry your question.")
            else:
                answer = {"role": "assistant", "content": result["answer"],
                          "passages": result["passages"], "total_s": result["total_s"]}
                st.session_state.messages.extend([
                    {"role": "user", "content": question}, answer,
                ])
                # Refresh controls and render the persisted conversation once;
                # preparation and inference are not repeated on this rerun.
                st.rerun()


if __name__ == "__main__":
    main()
