"""B1 with open-POM: the released open-POM 10-model ensemble, frozen, run by us (not the paper's
earlier output file), 138 odour-word probabilities per molecule, mean per mixture, cosine
distance, one line for all training studies. No training.

Shared steps in evaluation/B1/semantic.py.

Env:     mist (the embed step runs in env openpom if features/openpom.npz is missing)
Usage:   python evaluation/B1/eval_semantic_openpom.py            cross-validation only
         python evaluation/B1/eval_semantic_openpom.py --final    also score V50 and T46, once
         add --recompute-embeddings to rerun the pretrained model first
Writes:  results/eval_semantic_openpom/<date_time>[_final].{txt,json}
"""
import semantic

if __name__ == "__main__":
    semantic.run("eval_semantic_openpom", "openpom", __doc__)
