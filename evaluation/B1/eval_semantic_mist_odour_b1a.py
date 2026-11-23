"""B1a with the MIST odour model: B1 plus the paper's open-POM baseline steps. Frozen
dream2_data/mist_odour, 135 odour-word probabilities per molecule, each word quantile-transformed
to a normal over the 235 molecules, mean per mixture, scaled to length 1, Euclidean distance
(mixdist.mixture.paper_distance), one line for all training studies. No training.

B1a with open-POM is our open-POM baseline (evaluation/check_openpom_official.py).
Shared steps in evaluation/B1/semantic.py.

Env:     mist
Usage:   python evaluation/B1/eval_semantic_mist_odour_b1a.py            cross-validation only
         python evaluation/B1/eval_semantic_mist_odour_b1a.py --final    also score V50 and T46, once
Writes:  results/eval_semantic_mist_odour_b1a/<date_time>[_final].{txt,json}
"""
import semantic

if __name__ == "__main__":
    semantic.run("eval_semantic_mist_odour_b1a", "mist", __doc__, distance="paper")
