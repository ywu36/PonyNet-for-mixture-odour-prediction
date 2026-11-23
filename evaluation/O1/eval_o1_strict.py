"""O1 cross-validation, training data "strict": DREAM 2 distances only (plus GS-LF odour words of single molecules).

See o1_cv.py for the model, data and output. V50 and T46 only with --final (trains on all training pairs, scores both once; no CV).

Env:     mist (torch); needs features/openpom.npz and features/openpom_extra.npz
         (python evaluation/embed_openpom_extra.py, env openpom)
Usage:   python evaluation/O1/eval_o1_strict.py [--repeats 10] [--quick] | --final
Writes:  results/eval_o1_strict/<date_time>.{txt,json} (scores, training log, CV predictions)
"""
import o1_cv

if __name__ == "__main__":
    o1_cv.run("eval_o1_strict", "strict", __doc__)
