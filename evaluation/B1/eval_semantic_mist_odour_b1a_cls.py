"""B1a on the MIST odour model's embedding instead of its odour words. Frozen dream2_data/mist_odour,
its 512-d first-token (CLS) embedding per molecule (features/mist.npz, emb), each dimension
quantile-transformed to a normal over the 235 molecules, mean per mixture, scaled to length 1,
Euclidean distance (mixdist.mixture.paper_distance), one line for all training studies. No training.

Same as eval_semantic_mist_odour_b1a.py except the per-molecule input. Found in the B2-ft
cross-validation (evaluation/B2ft/eval_mist_mixture.py, "B1a readout of odour MIST frozen: CLS, mean").
Shared steps in evaluation/B1/semantic.py.

Env:     mist
Usage:   python evaluation/B1/eval_semantic_mist_odour_b1a_cls.py            cross-validation only
         python evaluation/B1/eval_semantic_mist_odour_b1a_cls.py --final    also score V50 and T46, once
Writes:  results/eval_semantic_mist_odour_b1a_cls/<date_time>[_final].{txt,json}
"""
import semantic

if __name__ == "__main__":
    semantic.run("eval_semantic_mist_odour_b1a_cls", "mist", __doc__, distance="paper", inputs="emb")
