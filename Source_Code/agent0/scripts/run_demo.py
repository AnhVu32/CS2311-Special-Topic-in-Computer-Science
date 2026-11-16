#!/usr/bin/env python3
"""Demo web (Gradio): nhập bài toán -> code Python + giải thích, có thể dùng KB few-shot RAG.

Chạy từ thư mục Source_Code:
    python -m agent0.scripts.run_demo                       # provider auto, model gemma3:12b
    python -m agent0.scripts.run_demo --provider ollama --model qwen2.5-coder:7b
Mặc định chỉ mở ở 127.0.0.1; thêm --host 0.0.0.0 nếu muốn máy khác trong mạng truy cập.
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from agent0.core.config import (
    DEFAULT_KB_PATH,
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_PROVIDERS,
    RuntimeConfig,
    describe_runtime,
    resolve_runtime,
)
from agent0.core.llm_client import chat_completion
from agent0.core.utils import extract_code_block, strip_think_blocks

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("demo")

DEFAULT_DEMO_MODEL = os.getenv("AGENT0_DEMO_MODEL", "gemma3:12b")


def find_kb_path(kb_path: Optional[str] = None) -> Optional[Path]:
    if kb_path:
        return Path(kb_path)
    if DEFAULT_KB_PATH.exists():
        return DEFAULT_KB_PATH
    if DEFAULT_OUTPUT_DIR.exists():
        for subdir in sorted(DEFAULT_OUTPUT_DIR.iterdir()):
            if subdir.is_dir() and subdir.name != "mbpp_seeded" and (subdir / "knowledge_base.jsonl").exists():
                return subdir / "knowledge_base.jsonl"
    return None


def load_retriever(kb_path: Optional[Path], embed_model: str, embed_base_url: str):
    if kb_path is None or not kb_path.exists():
        logger.warning("Không tìm thấy knowledge base -> demo chạy không có KB")
        return None
    from agent0.retrieval.retriever import KnowledgeRetriever

    try:
        retriever = KnowledgeRetriever(kb_path=str(kb_path), embed_model=embed_model, embed_base_url=embed_base_url)
        retriever.build_index()
    except Exception as exc:
        logger.warning("Không dựng được index KB (%s) -> demo chạy không có KB. Cần: ollama pull %s", exc, embed_model)
        return None
    logger.info("KB ready: %s entries từ %s", len(retriever.entries), kb_path)
    return retriever


def translate_to_english(runtime: RuntimeConfig, text: str) -> str:
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii < 3:
        return text
    try:
        result = chat_completion(runtime, [
            {"role": "system", "content": "Translate the following to English. Return ONLY the translation, nothing else."},
            {"role": "user", "content": text},
        ], temperature=0.0, max_tokens=256)
        return strip_think_blocks(result).strip()
    except Exception as exc:
        logger.warning("Translation failed: %s", exc)
        return text


def solve(runtime: RuntimeConfig, retriever, question: str, use_kb: bool, n_examples: int):
    if not question.strip():
        return "", "Vui lòng nhập câu hỏi.", ""

    query_en = translate_to_english(runtime, question)
    few_shot = ""
    retrieved_text = "Không dùng Knowledge Base."
    if use_kb and retriever:
        examples = retriever.query(query_en, n=int(n_examples))
        few_shot = retriever.format_few_shot(examples)
        if examples:
            retrieved_text = "\n\n".join(f"Sim={ex['similarity']:.4f}\n{ex['original_prompt']}" for ex in examples)
        else:
            retrieved_text = "Không tìm thấy ví dụ nào trong KB."
    elif use_kb:
        retrieved_text = "KB chưa sẵn sàng (thiếu file KB hoặc model embedding)."

    system_prompt = "You are a Python coding assistant. Return only runnable Python code, no markdown."
    if few_shot:
        user_prompt = f"Here are similar solved tasks:\n\n{few_shot}\n\nNow solve this task:\n{question}\n\nReturn only Python code."
    else:
        user_prompt = f"Task:\n{question}\n\nReturn only Python code."
    try:
        generated = chat_completion(runtime, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.1, max_tokens=1800)
        generated = strip_think_blocks(generated)
        code = extract_code_block(generated) or generated
    except Exception as exc:
        code = f"# Lỗi sinh code: {exc}"

    try:
        explanation = chat_completion(runtime, [
            {"role": "system", "content": "You are a coding tutor. Explain concisely in Vietnamese."},
            {"role": "user", "content": f"Explain this code briefly in Vietnamese:\n\n{code}"},
        ], temperature=0.1, max_tokens=500)
        explanation = strip_think_blocks(explanation)
    except Exception as exc:
        explanation = f"Lỗi sinh giải thích: {exc}"
    return code, explanation, retrieved_text


def build_app(runtime: RuntimeConfig, retriever):
    import gradio as gr

    def _solve(question, use_kb, n_examples):
        return solve(runtime, retriever, question, use_kb, n_examples)

    with gr.Blocks(title="Agent0 - Code Assistant") as app:
        gr.Markdown(f"# 🤖 Agent0 Coding Assistant (với Knowledge Base RAG)\nModel: `{runtime.model}` ({runtime.backend})")
        with gr.Row():
            with gr.Column(scale=1):
                q_in = gr.Textbox(lines=5, label="Nhập bài toán (Tiếng Việt/Anh)")
                use_kb = gr.Checkbox(label="Sử dụng Knowledge Base (Few-shot RAG)", value=retriever is not None)
                n_ex = gr.Slider(1, 5, value=3, step=1, label="Số lượng ví dụ (Top-K)")
                btn = gr.Button("Giải bài toán", variant="primary")
                ret_out = gr.Textbox(lines=10, label="Các bài toán tương tự được tìm thấy trong KB")
            with gr.Column(scale=2):
                code_out = gr.Code(language="python", label="Python Code")
                exp_out = gr.Markdown(label="Giải thích")
        btn.click(fn=_solve, inputs=[q_in, use_kb, n_ex], outputs=[code_out, exp_out, ret_out])
    return app


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Agent0 demo web (Gradio)")
    parser.add_argument("--provider", default="auto", choices=SUPPORTED_PROVIDERS)
    parser.add_argument("--model", default=DEFAULT_DEMO_MODEL)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--kb_path", default=None, help=f"Mặc định: {DEFAULT_KB_PATH}")
    parser.add_argument("--embed_model", default="mxbai-embed-large")
    parser.add_argument("--embed_base_url", default="http://localhost:11434")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    runtime = resolve_runtime(args.provider, args.model, args.base_url, args.api_key)
    logger.info("Runtime: %s", describe_runtime(runtime))
    retriever = load_retriever(find_kb_path(args.kb_path), args.embed_model, args.embed_base_url)
    build_app(runtime, retriever).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
