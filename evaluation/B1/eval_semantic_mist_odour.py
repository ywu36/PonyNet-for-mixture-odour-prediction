"""B1 with the MIST odour model: frozen dream2_data/mist_odour, 135 odour-word probabilities per
molecule, mean per mixture, cosine distance, one line for all training studies. No training.

Shared steps in evaluation/B1/semantic.py.

Env:     mist
Usage:   python evaluation/B1/eval_semantic_mist_odour.py            cross-validation only
         python evaluation/B1/eval_semantic_mist_odour.py --final    also score V50 and T46, once
         add --recompute-embeddings to rerun the pretrained model first
Writes:  results/eval_semantic_mist_odour/<date_time>[_final].{txt,json}
"""
import semantic

if __name__ == "__main__":
    semantic.run("eval_semantic_mist_odour", "mist", __doc__)
