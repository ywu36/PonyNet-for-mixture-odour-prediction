"""O1 cross-validation, training data "strict": DREAM 2 distances only (plus GS-LF odour words of single molecules); encoder open-POM (frozen).

With cross-fitted calibration (xcal): after training, a line y = a * prediction + c fitted on
out-of-fold predictions inside the training fold (5 inner zero-overlap splits; model frozen),
so the level and spread on new mixtures match the human scale. Only training-fold labels.

2 x 5 folds by default (identical to the first 2 repeats of the 10-repeat runs). See o1_cv.py for the
model, data and output. V50 and T46 only with --final (trains on all training pairs, scores both
once; no CV).

Env:     mist (torch); needs features/openpom.npz and features/openpom_extra.npz
Usage:   python evaluation/O1/eval_o1_strict_xcal.py [--repeats 2] [--quick] | --final
Writes:  results/eval_o1_strict_xcal/<date_time>.{txt,json}
"""
import o1_cv

if __name__ == "__main__":
    o1_cv.run("eval_o1_strict_xcal", "strict", __doc__, xcal=True, encoder="openpom", repeats=2)
