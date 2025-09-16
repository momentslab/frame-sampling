import evaluate
from pycocoevalcap.cider.cider import Cider
import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
bertscore_metric = evaluate.load("bertscore")


def compute_scores_from_df(df,metrics):
    references = df["description"].tolist()
    hypotheses = df["Prediction"].tolist()
    f = compute_all_scores(references, hypotheses, metrics)
    return f

def plt_fig(gf,path_destination):
    fig, ax = plt.subplots(figsize=(12, 6))
    gf.T.plot(kind='bar', ax=ax)

    ax.set_yticks([i / 10 for i in range(11)])

    plt.title("Evaluation Metrics for Vision-Language Models")
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.xticks(rotation=45)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(path_destination)

def compute_all_scores(references, hypotheses, metrics):
    scores = {}

    if "BLEU-1" in metrics or "BLEU-2" in metrics or "BLEU-3" in metrics or "BLEU-4" in metrics:
        bleu_vals = bleu_scores(references, hypotheses)
        scores.update({
            "BLEU-1": bleu_vals[0],
            "BLEU-2": bleu_vals[1],
            "BLEU-3": bleu_vals[2],
            "BLEU-4": bleu_vals[3],
        })

    if "ROUGE-L" in metrics:
        scores["ROUGE-L"] = rouge_score(references, hypotheses)

    if "METEOR" in metrics:
        scores["METEOR"] = meteor_score(references, hypotheses)

    if "CIDEr" in metrics:
        scores["CIDEr"] = cider_score(references, hypotheses)

    if "BERTScore" in metrics:
        scores["BERTScore"] = bertscore_f1(references, hypotheses)

    return scores

def bleu_scores(references, hypotheses):
    ref_flat = [[ref] for ref in references]
    scores = []
    bleu_evaluator = evaluate.load("bleu", module_type="metric")
    for n in range(1, 5):
        result = bleu_evaluator.compute(
            predictions=hypotheses,
            references=ref_flat,
            max_order=n
        )
        scores.append(result["bleu"])
    return scores

def rouge_score(references, hypotheses):
    result = rouge_metric.compute(
        predictions=hypotheses,
        references=references,
        rouge_types=["rougeL"]
    )
    return float(result["rougeL"])

def meteor_score(references, hypotheses):
    result = meteor_metric.compute(
        predictions=hypotheses,
        references=references
    )
    return float(result["meteor"])

def cider_score(references, hypotheses):
    cider_scorer = Cider()
    formatted_refs = {i: [ref] for i, ref in enumerate(references)}
    formatted_hyps = {i: [hyp] for i, hyp in enumerate(hypotheses)}
    score, _ = cider_scorer.compute_score(formatted_refs, formatted_hyps)
    return score

def bertscore_f1(references, hypotheses, lang="en", batch_size=64):
    result = bertscore_metric.compute(
        predictions=hypotheses,
        references=references,
        lang=lang,
        batch_size=batch_size,
        device=device  # CUDA if available
    )
    avg_f1 = sum(result["f1"]) / len(result["f1"])
    return avg_f1
