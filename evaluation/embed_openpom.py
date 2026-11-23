"""Compute per-molecule open-POM features with the authors' pretrained ensemble and cache them.

Runs the 10 released open-POM models (graph neural networks) on the 235 molecules and averages
them: 138 odour-word probabilities and a 256-d embedding per molecule (each member's outputs are
kept too). Only embeds; checks are in check_openpom_official.py.

Env:     openpom (DeepChem + DGL; CPU build of DGL, so it runs on the CPU, about a minute)
Usage:   python evaluation/embed_openpom.py
Needs:   the openpom repo with real checkpoints, found via mixdist.data.locate("openpom")
         (e.g. OSMO_DATA=~/.cache/osmo-data-mirror, which holds external/openpom)
Writes:  features/openpom.npz, results/embed_openpom/<date_time>.{txt,json}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import features, openpom  # noqa: E402
from mixdist.device import pick_device  # noqa: E402
from mixdist.report import Report  # noqa: E402


def main():
    rep = Report("embed_openpom")
    rep.line(f"open-POM repo {openpom.repo()}")
    arrays = openpom.compute(pick_device(), progress=rep.line)
    f = features.save("openpom", arrays)
    fs = features.load("openpom")
    rep.line(fs.describe())
    rep.line(f"wrote {f}")

    rep.line("check it with: python evaluation/check_openpom_official.py")
    rep.save()


if __name__ == "__main__":
    main()
