"""O1 cross-validation, training data "joint": pretrain on the GoodScents blend pairs, then DREAM 2 distances with the pair loss kept on.

See o1_cv.py for the model, data and output. V50 and T46 only with --final (trains on all training pairs, scores both once; no CV).

Env:     mist (torch); needs features/openpom.npz and features/openpom_extra.npz
         (python evaluation/embed_openpom_extra.py, env openpom)
Usage:   python evaluation/O1/eval_o1_joint.py [--repeats 10] [--quick] | --final
Writes:  results/eval_o1_joint/<date_time>.{txt,json} (scores, training log, CV predictions)
"""
import o1_cv

if __name__ == "__main__":
    o1_cv.run("eval_o1_joint", "joint", __doc__)
