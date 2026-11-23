"""O1 cross-validation, training data "strict" (DREAM 2 distances + GS-LF odour words), with inner
calibration: in each training fold, whole mixtures are held out once more; the model is trained on
the rest, stopped at the step with the lowest error on the held-out part, and a line
prediction -> label is fitted there and applied to the outer test fold. Only training-fold labels
are used. Fixes the CV bias of eval_o1_strict.py (predictions too high on new mixtures).

See o1_cv.py for the model, data and output. V50 and T46 only with --final (trains on all training pairs, scores both once; no CV).

Env:     mist (torch); needs features/openpom.npz and features/openpom_extra.npz
Usage:   python evaluation/O1/eval_o1_strict_cal.py [--repeats 10] [--quick] | --final
Writes:  results/eval_o1_strict_cal/<date_time>.{txt,json}
"""
import o1_cv

if __name__ == "__main__":
    o1_cv.run("eval_o1_strict_cal", "strict", __doc__, inner_calibration=True)
