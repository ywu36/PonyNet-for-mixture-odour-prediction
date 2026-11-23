"""O1.1 cross-validation (see o10_cv.py, variant "o11"): open-POM's 138 frozen odour words as receptors;
molecules compete for each receptor's sites, binding by their rank for that word:
r_w(A) = sum_m e(m, w) / (1 + s_w * sum_m q(m, w)); 138 binding strengths s_w + a line; B1a at s_w = 0.

V50 and T46 only with --final (trains on all training pairs, scores both once; no CV).

Env:     mist (torch); needs features/openpom.npz
Usage:   python evaluation/O1/eval_o11.py [--repeats 10] [--quick] [--live] | --final
Writes:  results/eval_o11/<date_time>.{txt,json} (scores, training log, CV predictions)
"""
import o10_cv

if __name__ == "__main__":
    o10_cv.run("eval_o11", "o11", __doc__)
