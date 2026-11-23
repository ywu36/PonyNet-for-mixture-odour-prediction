"""Finding 3 I2/I3: B1a + discrete pull/push terms (both / one side / frequency gap) on open-POM's 138 odour words (quantile score > 1).

See units_cv.py for the features, checks and output. Training pairs only (CV); no V50/T46.

Env:     mist
Usage:   python evaluation/Substructure/eval_units_words.py [--perms 50] [--repeats 10] [--quick]
Writes:  results/eval_units_words/<date_time>.{txt,json}
"""
import units_cv

if __name__ == "__main__":
    units_cv.run("eval_units_words", ['words'], __doc__)
