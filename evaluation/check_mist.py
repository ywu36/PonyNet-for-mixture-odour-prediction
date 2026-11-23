"""Check the MIST embeddings end to end: inputs, odour outputs, embedding vectors, agreement.

Sections 0-4 are the standard checks of mixdist.checks (same for every model):
  0 embeddings computed (runs embed_mist.py in the mist env if missing)
  1 the Kekulé SMILES MIST was given are the same molecules as our data
  2 MIST's 135 odour-word probabilities vs real GS-LF labels of our molecules (AUROC, shuffled control)
  3 MIST's 512-d embedding: linear probe to the GS-LF labels (AUROC, shuffled control)
  4 agreement with open-POM on the shared odour words
MIST publishes no reference outputs or scores (its model card names AUROC but gives no number),
so there is no model-specific comparison as in check_openpom_official.py.

Env:     mist
Usage:   python evaluation/check_mist.py [--recompute-embeddings]
Writes:  results/check_mist/<date_time>.{txt,json}
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import checks  # noqa: E402
from mixdist.report import Report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recompute-embeddings", action="store_true", help="rerun MIST first")
    args = ap.parse_args()
    rep = Report("check_mist", vars(args))
    checks.run_standard_checks(rep, "mist", recompute=args.recompute_embeddings)
    rep.section("5. MIST-specific")
    rep.line("none: MIST publishes no reference predictions or scores to compare with")
    rep.save()


if __name__ == "__main__":
    main()
