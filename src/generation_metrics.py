# src/generation_metrics.py

import json
from pathlib import Path

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# --------- Load Test Data ---------

def load_eval_data(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)

# --------- BLEU ---------

def compute_bleu(reference: str, prediction: str):
    smooth = SmoothingFunction().method1
    return sentence_bleu(
        [reference.split()],
        prediction.split(),
        smoothing_function=smooth
    )

# --------- ROUGE-L ---------

def compute_rouge_l(reference: str, prediction: str):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(reference, prediction)
    return score["rougeL"].fmeasure

# --------- Main Eval Runner ---------

def run_generation_eval():
    json_path = Path(__file__).resolve().parents[1] / "data" / "test" / "test_queries_generation.json"
    test_data = load_eval_data(json_path)

    print("\n==============================")
    print("📝 Running GENERATION Evaluation")
    print("==============================")

    for item in test_data:
        reference = item["reference"]
        prediction = item["model_answer"]

        print(f"\nQuestion: {item['question']}")
        print("Reference:", reference)
        print("Model Output:", prediction)

        bleu = compute_bleu(reference, prediction)
        rouge_l = compute_rouge_l(reference, prediction)

        print(f"BLEU Score: {bleu:.4f}")
        print(f"ROUGE-L:   {rouge_l:.4f}")

if __name__ == "__main__":
    run_generation_eval()
