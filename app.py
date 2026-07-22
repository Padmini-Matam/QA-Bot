"""
QueryVault - Gradio Web Interface
Run this file to launch the QA bot in your browser.

Usage:
    python app.py
"""

import gradio as gr
from dotenv import load_dotenv
from src.qabot import process_pdf_and_query

# Load environment variables from .env file
load_dotenv()


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

def answer_question(pdf_file, question: str) -> str:
    """
    Wrapper called by Gradio on each user interaction.

    Args:
        pdf_file: Gradio file object (has a .name attribute with the temp path).
        question: User's question string.

    Returns:
        Answer string to display in the output box.
    """
    if pdf_file is None:
        return "⚠️  Please upload a PDF document before asking a question."

    file_path = pdf_file.name  # Gradio stores uploaded files at a temp path
    return process_pdf_and_query(file_path, question)


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="QueryVault – PDF QA Bot"
) as demo:

    gr.Markdown(
        """
        # 📄 QueryVault — PDF Question Answering Bot
        Upload a PDF document, ask any question about its content, and get an
        intelligent answer powered by **Groq API (Llama-3.3 70B)** and **LangChain RAG**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="📁 Upload PDF",
                file_types=[".pdf"],
                type="filepath",
            )
            question_input = gr.Textbox(
                label="❓ Your Question",
                placeholder="e.g. What are the main findings of this document?",
                lines=3,
            )
            submit_btn = gr.Button("🔍 Get Answer", variant="primary")

        with gr.Column(scale=1):
            answer_output = gr.Textbox(
                label="💡 Answer",
                lines=12,
                interactive=False,
            )

    submit_btn.click(
        fn=answer_question,
        inputs=[pdf_input, question_input],
        outputs=answer_output,
    )

    gr.Markdown(
        """
        ---
        **How it works:**
        1. Your PDF is parsed and split into chunks.
        2. Each chunk is converted to a semantic embedding via HuggingFace Local Embeddings (`all-MiniLM-L6-v2`).
        3. The most relevant chunks are retrieved using ChromaDB.
        4. Llama-3.3 70B reads the context and generates a precise answer.
        """
    )


if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())
