"""B1a with open-POM = our open-POM baseline: released open-POM ensemble, 138 odour-word
probabilities per molecule, quantile per word, mean per mixture, length 1, Euclidean, one line
for all training pairs. Same recipe as the paper's notebook (evaluation/check_openpom_official.py),
which fits on 817 of the 820 pairs (no Metamer); predictions agree to 0.001.

Shared steps in evaluation/B1/semantic.py.

Env:     mist
Usage:   python evaluation/B1/eval_semantic_openpom_b1a.py            cross-validation only
         python evaluation/B1/eval_semantic_openpom_b1a.py --final    also score V50 and T46, once
Writes:  results/eval_semantic_openpom_b1a/<date_time>[_final].{txt,json}
"""
import semantic

if __name__ == "__main__":
    semantic.run("eval_semantic_openpom_b1a", "openpom", __doc__, calibration="shared", distance="paper")
