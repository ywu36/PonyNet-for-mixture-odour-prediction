"""mixdist.teams: helpers for rerunning a team's code (no team code is run here).

Run with pytest, or directly:  python tests/test_teams.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mixdist import data, teams  # noqa: E402


def test_read_submission_follows_pair_order():
    pairs = [data.Pair("Test", "2", "18", frozenset(), frozenset(), 0.5),
             data.Pair("Test", "1", "17", frozenset(), frozenset(), 0.4)]
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "form.csv"
        f.write_text("Dataset,Mixture_1,Mixture_2,Predicted_Experimental_Values\n"
                     "CGm01  ,1,17,0.25\nCGm02  ,2,18,0.75\n")
        assert teams.read_submission(f, pairs).tolist() == [0.75, 0.25]


def test_fresh_copy_replaces_old_files_and_skips_git():
    with tempfile.TemporaryDirectory() as d:
        src, work = Path(d) / "src", Path(d) / "runs" / "team"
        (src / ".git").mkdir(parents=True)
        (src / "code.py").write_text("x = 1\n")
        work.mkdir(parents=True)
        (work / "stale.txt").write_text("old")
        teams.fresh_copy(src, work)
        assert sorted(p.name for p in work.iterdir()) == ["code.py"]


def test_free_gpu_is_an_index_or_none():
    gpu = teams.free_gpu()
    assert gpu is None or gpu.isdigit()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
