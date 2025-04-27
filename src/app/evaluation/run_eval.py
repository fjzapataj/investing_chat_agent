import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
src_dir = project_root / "src"

sys.path.insert(0, str(src_dir))

import os
import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain

from app.graphs.state_machine import graph

def get_full_response(user_input: str, config: dict) -> str:
    respuesta = ""
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
    ):
        for value in event.values():
            respuesta += value["messages"][-1].content
    return respuesta


def main():
    load_dotenv()

    # Configuración del LLM y del evaluador
    config = {"configurable": {"thread_id": "eval"}}
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    eval_chain = QAEvalChain.from_llm(llm)

    # Cargar el dataset desde src/tests/eval_dataset.json
    tests_dir = src_dir / "tests"
    eval_file = tests_dir / "eval_dataset.json"
    with open(eval_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    examples = []
    predictions = []

    # Iterar sobre cada pregunta y obtener la predicción del bot
    for item in dataset:
        q, ref = item["question"], item["answer"]
        pred = get_full_response(q, config)

        examples.append({"query": q, "answer": ref})
        predictions.append({"result": pred})

        print(f"\nPregunta: {q}")
        print(f"→ Bot: {pred}")
        print(f"→ Ref: {ref}")

    print("\n[*] Ejecutando QAEvalChain…")
    graded = eval_chain.evaluate(examples, predictions)

    # Resultados en src/app/evaluation/run_eval_results.json
    output = []
    for item, grade, pred in zip(dataset, graded, predictions):
        output.append({
            "question": item["question"],
            "reference_answer": item["answer"],
            "bot_answer": pred["result"],
            "evaluation": grade.get("results", grade)
        })

    result_file = current_file.parent / "run_eval_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[✔] Evaluación completada. Revisa {result_file}")


if __name__ == "__main__":
    main()