"""
Step 3 - RAGAS Evaluation.

Runs 50 QA pairs through both prompt versions, evaluates with RAGAS,
prints a comparison table, and saves data/ragas_report.json plus an
evidence copy.
"""
import json
import sys
import types
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import config

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# RAGAS 0.4.x imports this legacy LangChain module even when Vertex AI is not
# used. Newer langchain-community builds no longer ship it, so provide a tiny
# compatibility shim before importing ragas.
vertexai_module = "langchain_community.chat_models.vertexai"
if vertexai_module not in sys.modules:
    shim = types.ModuleType(vertexai_module)

    class ChatVertexAI:
        pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[vertexai_module] = shim

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from qa_pairs import QA_PAIRS
from utils.data_loader import build_vectorstore, load_knowledge_base, split_text
from utils.llm_factory import get_embeddings, get_llm


SYSTEM_V1 = (
    "You are a concise AI study assistant. Answer in 2-4 clear sentences using only the provided context. "
    "If the context does not contain the answer, say you cannot find the information.\n\n"
    "Context:\n{context}"
)

SYSTEM_V2 = (
    "You are an expert AI systems tutor. Read the context carefully, identify the relevant facts, "
    "and answer in a structured 3-5 sentence explanation. Mention the key concept first, then add "
    "supporting detail from the context. Do not invent facts outside the context.\n\n"
    "Context:\n{context}"
)

PROMPTS = {
    "v1": ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")]),
    "v2": ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")]),
}

PROJECT_ROOT = Path(__file__).parent.parent
RAG_OUTPUTS_PATH = PROJECT_ROOT / "data" / "ragas_outputs.json"


def setup_vectorstore():
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    return build_vectorstore(chunks, embeddings)


def run_rag(retriever, llm, prompt, question: str) -> dict:
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    context_text = "\n\n".join(contexts)
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context_text, "question": question}
    )
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list[dict]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    prompt = PROMPTS[prompt_version]
    results = []

    print(f"\nRunning 50 QA pairs with prompt {prompt_version} ...")
    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])
        results.append(
            {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": out["answer"],
                "contexts": out["contexts"],
            }
        )
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


def build_ragas_dataset(rag_results: list[dict]) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=row["question"],
            response=row["answer"],
            retrieved_contexts=row["contexts"],
            reference=row["reference"],
        )
        for row in rag_results
    ]
    return EvaluationDataset(samples=samples)


def _metric_values(result, key: str) -> list[float]:
    raw = result[key]
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return [float(v) for v in raw if v is not None and not np.isnan(float(v))]


def run_ragas_eval(rag_results: list[dict], version: str) -> dict:
    print(f"\nEvaluating RAGAS for prompt {version} ...")
    dataset = build_ragas_dataset(rag_results)
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=RunConfig(timeout=240, max_retries=6, max_wait=120, max_workers=1),
        batch_size=1,
    )

    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        values = _metric_values(result, key)
        scores[key] = float(np.mean(values)) if values else 0.0

    print(f"\nRAGAS results - Prompt {version.upper()}:")
    for key, value in scores.items():
        marker = " * target met" if key == "faithfulness" and value >= 0.8 else ""
        print(f"  {key:30s}: {value:.4f}{marker}")

    return scores


def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    if RAG_OUTPUTS_PATH.exists():
        cached = json.loads(RAG_OUTPUTS_PATH.read_text(encoding="utf-8"))
        v1_results = cached["v1_results"]
        v2_results = cached["v2_results"]
        print(f"Loaded cached RAG outputs from {RAG_OUTPUTS_PATH}")
    else:
        vectorstore = setup_vectorstore()
        v1_results = collect_rag_outputs(vectorstore, "v1")
        v2_results = collect_rag_outputs(vectorstore, "v2")
        RAG_OUTPUTS_PATH.write_text(
            json.dumps(
                {"v1_results": v1_results, "v2_results": v2_results},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Saved RAG outputs to {RAG_OUTPUTS_PATH}")

    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1 = v1_scores[metric]
        s2 = v2_scores[metric]
        winner = "<- V1" if s1 > s2 else "<- V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    target_met = best_faith >= 0.8
    if target_met:
        print(f"\nTarget met: faithfulness = {best_faith:.4f} >= 0.8")
    else:
        print(f"\nTarget not met: faithfulness = {best_faith:.4f} < 0.8")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": target_met,
    }
    data_report_path = PROJECT_ROOT / "data" / "ragas_report.json"
    evidence_report_path = PROJECT_ROOT / "evidence" / "03_ragas_report.json"
    data_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    evidence_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to {data_report_path}")
    print(f"Copied report to {evidence_report_path}")


if __name__ == "__main__":
    main()
