import os
import json
import re
import string
import collections
import numpy as np

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_results_file = os.path.join(base_dir, 'results_raw.json')
    summary_json_file = os.path.join(base_dir, 'results_summary.json')
    summary_md_file = os.path.join(base_dir, 'results_summary.md')

    if not os.path.exists(raw_results_file):
        print(f"Error: {raw_results_file} not found. Run run_evaluation.py first.")
        return

    with open(raw_results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    if not results:
        print("No results found to evaluate.")
        return

    print("Loading SentenceTransformer model for semantic similarity...")
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("Error: sentence-transformers is not installed. Please run pip install sentence-transformers.")
        return

    total_em = 0
    total_f1 = 0
    total_sim = 0
    latencies = []

    print(f"Computing metrics for {len(results)} predictions...")

    for res in results:
        gold = res['ground_truth']
        pred = res['prediction']
        
        # EM and F1
        em = compute_exact(gold, pred)
        f1 = compute_f1(gold, pred)
        
        # Semantic Similarity
        gold_emb = model.encode(gold, convert_to_tensor=True)
        pred_emb = model.encode(pred, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(gold_emb, pred_emb).item()

        res['metrics'] = {
            'exact_match': em,
            'f1_score': f1,
            'semantic_similarity': sim
        }
        
        total_em += em
        total_f1 += f1
        total_sim += sim
        latencies.append(res['latency_seconds'])

    num_samples = len(results)
    avg_em = total_em / num_samples
    avg_f1 = total_f1 / num_samples
    avg_sim = total_sim / num_samples
    
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    summary = {
        "num_samples": num_samples,
        "exact_match": avg_em,
        "f1_score": avg_f1,
        "semantic_similarity": avg_sim,
        "latency_avg_sec": avg_latency,
        "latency_p95_sec": p95_latency
    }

    # Save summary json
    with open(summary_json_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
        
    # Write summary md
    md_content = "# QueryVault Evaluation Results\n\n"
    md_content += "## Test Set Composition & Limitations\n"
    md_content += "> [!WARNING]\n"
    md_content += "> The `academic_paper.pdf` ('Attention Is All You Need') is a widely-known text that the Llama-3.3 70B model has likely memorized during pre-training. Its scores should be interpreted as an upper bound partially influenced by model prior knowledge.\n"
    md_content += "> The `technical_manual.pdf` (a recently published 2026 arXiv paper) and `legal_policy.pdf` better isolate actual retrieval quality by minimizing pre-training contamination.\n\n"
    md_content += f"**Total Samples Evaluated**: {num_samples}\n\n"
    md_content += "| Metric | Score |\n"
    md_content += "|---|---|\n"
    md_content += f"| Exact Match (EM) | {avg_em:.4f} |\n"
    md_content += f"| Token F1 Score | {avg_f1:.4f} |\n"
    md_content += f"| Semantic Similarity | {avg_sim:.4f} |\n"
    md_content += f"| Avg Latency | {avg_latency:.2f}s |\n"
    md_content += f"| P95 Latency | {p95_latency:.2f}s |\n"

    with open(summary_md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print("\n" + md_content)
    print(f"Metrics saved to {summary_json_file} and {summary_md_file}")

if __name__ == "__main__":
    main()
