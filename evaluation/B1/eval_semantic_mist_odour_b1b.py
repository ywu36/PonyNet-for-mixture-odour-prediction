"""B1b with the MIST odour model: B1a (eval_semantic_mist_odour_b1a.py) with one calibration
line for pairs whose mixtures share no molecule and one for pairs that share at least one
(mixdist.evaluate, scheme 'overlap'). Every V50 and T46 pair shares no molecule, so both use
the first line.

Shared steps in evaluation/B1/semantic.py.

Env:     mist
Usage:   python evaluation/B1/eval_semantic_mist_odour_b1b.py            cross-validation only
         python evaluation/B1/eval_semantic_mist_odour_b1b.py --final    also score V50 and T46, once
Writes:  results/eval_semantic_mist_odour_b1b/<date_time>[_final].{txt,json}
"""
import semantic

if __name__ == "__main__":
    semantic.run("eval_semantic_mist_odour_b1b", "mist", __doc__, calibration="overlap", distance="paper")
