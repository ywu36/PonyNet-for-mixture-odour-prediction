"""O1.0 cross-validation, variant "no_unit" (see o10_cv.py): competition on open-POM's 138 frozen odour words.

V50 and T46 only with --final (trains on all training pairs, scores both once; no CV).

Env:     mist (torch); needs features/openpom.npz
Usage:   python evaluation/O1/eval_o10_no_unit.py [--repeats 10] [--quick] [--live] | --final
Writes:  results/eval_o10_no_unit/<date_time>.{txt,json} (scores, training log, CV predictions)
"""
import o10_cv

if __name__ == "__main__":
    o10_cv.run("eval_o10_no_unit", "no_unit", __doc__)
