"""O1.0 cross-validation, variant "weights_only" (see o10_cv.py): competition on open-POM's 138 frozen odour words.

V50 and T46 only with --final (trains on all training pairs, scores both once; no CV).

Env:     mist (torch); needs features/openpom.npz
Usage:   python evaluation/O1/eval_o10_weights_only.py [--repeats 10] [--quick] [--live] | --final
Writes:  results/eval_o10_weights_only/<date_time>.{txt,json} (scores, training log, CV predictions)
"""
import o10_cv

if __name__ == "__main__":
    o10_cv.run("eval_o10_weights_only", "weights_only", __doc__)
