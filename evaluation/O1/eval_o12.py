"""O1.2 cross-validation (see o10_cv.py, variant "o12"): open-POM's 138 frozen odour words; within a mixture, the molecules strongest
on a word dominate it: r_w(A) = sum_m softmax_m(tau_w e(m, w)) e(m, w); 138 tau_w + a line; B1a at tau_w = 0.

V50 and T46 only with --final (trains on all training pairs, scores both once; no CV).

Env:     mist (torch); needs features/openpom.npz
Usage:   python evaluation/O1/eval_o12.py [--repeats 10] [--quick] [--live] | --final
Writes:  results/eval_o12/<date_time>.{txt,json} (scores, training log, CV predictions)
"""
import o10_cv

if __name__ == "__main__":
    o10_cv.run("eval_o12", "o12", __doc__)
