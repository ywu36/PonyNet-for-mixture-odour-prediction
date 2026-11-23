"""O1 cross-validation, training data "pairs": pretrain on the GoodScents blend pairs only; DREAM 2 sets only a line on the receptor distance; encoder MIST (frozen).

2 x 5 folds by default (identical to the first 2 repeats of the 10-repeat runs). See o1_cv.py for the
model, data and output. V50 and T46 only with --final (trains on all training pairs, scores both
once; no CV).

Env:     mist (torch); needs features/mist.npz and features/mist_extra.npz (python evaluation/embed_mist_extra.py)
Usage:   python evaluation/O1/eval_o1_pairs_mist.py [--repeats 2] [--quick] | --final
Writes:  results/eval_o1_pairs_mist/<date_time>.{txt,json}
"""
import o1_cv

if __name__ == "__main__":
    o1_cv.run("eval_o1_pairs_mist", "pairs", __doc__, xcal=False, encoder="mist", repeats=2)
