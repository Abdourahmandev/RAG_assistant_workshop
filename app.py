"""Run the complete document assistant without opening a notebook."""

import argparse
from pathlib import Path
import sys

from rag.application import PROJECT_DIR, inspect_pdf, prepare_assistant, safe_error
from rag.evaluation import load_test_cases


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Prepare a PDF and chat with the RAG assistant.")
    command.add_argument("--pdf", type=Path, default=PROJECT_DIR / "A-6.002.pdf", help="PDF to index")
    command.add_argument("--chunk-size", type=int, default=80)
    command.add_argument("--overlap", type=int, default=15)
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--k", type=int, default=3, help="Number of passages per answer")
    command.add_argument("--force-rebuild", action="store_true", help="Recompute embeddings in a new cache generation")
    action = command.add_mutually_exclusive_group()
    action.add_argument("--question", help="Ask one question, then exit")
    action.add_argument("--prepare-only", action="store_true", help="Prepare the index, then exit")
    action.add_argument("--evaluate", action="store_true", help="Run the seven workshop questions and save the metrics")
    action.add_argument("--inspect", action="store_true", help="Count pages and passages without calling Hugging Face")
    return command


def print_answer(result: dict) -> None:
    print("\nAssistant :", result["answer"])
    print("Retrieved sources:")
    for passage in result["passages"]:
        print(f"  [{passage['id']}] {passage['title']}")
    print(f"Response time: {result['total_s']:.1f} s\n")


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    assistant = None
    try:
        if arguments.k < 1 or arguments.batch_size < 1:
            raise ValueError("--k and --batch-size must be positive.")
        plan = inspect_pdf(arguments.pdf, arguments.chunk_size, arguments.overlap)
        print(f"Document : {plan.path.name} · {len(plan.documents)} pages · {len(plan.chunks)} passages", flush=True)
        if arguments.inspect:
            return 0
        if arguments.question is not None and not (0 < len(arguments.question.strip()) <= 2000):
            raise ValueError("The question must contain between 1 and 2,000 characters.")
        if arguments.evaluate:
            load_test_cases(plan.chunks)  # Fail before paying for embeddings if evidence changed.
        assistant = prepare_assistant(
            plan, batch_size=arguments.batch_size, force_rebuild=arguments.force_rebuild,
            database_dir=PROJECT_DIR / "base_vectorielle" / "cli",
            progress=lambda event: print(event.message, flush=True),
        )
        if arguments.prepare_only:
            return 0
        if arguments.evaluate:
            json_path, csv_path, kpis = assistant.evaluate(PROJECT_DIR / "resultats", arguments.k)
            for label, value in kpis.items():
                print(f"{label}: {value}")
            print(f"Reports:\n{json_path}\n{csv_path}")
            return 0 if kpis["Technical success"] == "7/7" else 1
        if arguments.question:
            print_answer(assistant.answer(arguments.question, k=arguments.k))
            return 0
        print("Ask your questions. /new clears the conversation; /quit ends the session.")
        history = []
        while True:
            try:
                question = input("You: ").strip()
            except EOFError:
                return 0
            if question.lower() in ("/quit", "quit", "exit"):
                return 0
            if question.lower() == "/new":
                history.clear()
                print("New conversation.")
                continue
            if not question:
                continue
            try:
                result = assistant.answer(question, history, arguments.k)
            except Exception as exc:
                print("Error: " + safe_error(exc), file=sys.stderr)
                continue
            print_answer(result)
            history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": result["answer"]},
            ])
            history = history[-6:]
    except KeyboardInterrupt:
        print("\nStopped. Completed batches remain cached for the next run.")
        return 130
    except Exception as exc:
        print("Error: " + safe_error(exc), file=sys.stderr)
        return 1
    finally:
        if assistant is not None:
            assistant.close()


if __name__ == "__main__":
    raise SystemExit(main())
