"""O1 cross-validation, training data "joint": pretrain on the GoodScents blend pairs, then DREAM 2 distances with the pair loss kept on; encoder MIST (frozen).

2 x 5 folds by default (identical to the first 2 repeats of the 10-repeat runs). See o1_cv.py for the
model, data and output. V50 and T46 only with --final (trains on all training pairs, scores both
once; no CV).

Env:     mist (torch); needs features/mist.npz and features/mist_extra.npz (python evaluation/embed_mist_extra.py)
Usage:   python evaluation/O1/eval_o1_joint_mist.py [--repeats 2] [--quick] | --final
Writes:  results/eval_o1_joint_mist/<date_time>.{txt,json}
"""
import o1_cv

if __name__ == "__main__":
    o1_cv.run("eval_o1_joint_mist", "joint", __doc__, xcal=False, encoder="mist", repeats=2)
