"""Standard output for every script: the same header, section and table layout on screen, and a
saved copy of each run under results/<script>/<date_time>[_<tag>].txt plus a .json with the
same content in machine-readable form.

    rep = Report("eval_semantic_mist_odour", args={"model": "mist"}, tag="final")
    rep.section("Cross-validation on the training set")
    rep.scores(rows)                  # rows: [(label, metrics.score(...) dict), ...]
    rep.table(["fold", "r"], [["0", 0.41], ...])
    rep.save()

Numbers are printed with 3 decimals unless a table cell is already a string.
Standard numpy and the standard library only, so it works in every conda environment.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WIDTH = 96


def results_dir() -> Path:
    return Path(os.environ.get("OSMO_RESULTS", REPO / "results"))


def _git_state() -> str:
    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    try:
        sha = git("rev-parse", "--short", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no", "--", "mixdist", "evaluation", "paper"))
    except OSError:
        return "git not available"
    if not sha:
        return "git: no commits yet"
    return f"git {sha}" + (" + uncommitted changes" if dirty else "")


def _data_location() -> str:
    try:
        from .data import locate
        return str(locate("pnas_repo"))
    except Exception as e:                                   # report must still work without data
        return f"not found ({e.__class__.__name__})"


def _plain(x):
    """JSON-safe copy (numpy scalars and tuples to plain floats and lists)."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return x


def _cell(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, float):
        return "nan" if x != x else f"{x:.3f}"
    return str(x)


class Report:
    def __init__(self, script: str, args: dict | None = None, tag: str = ""):
        self.script, self.args, self.tag = script, dict(args or {}), tag
        self.time = _dt.datetime.now()
        self.info = {
            "env": os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name,
            "python": f"{sys.executable} ({sys.version.split()[0]})",
            "data": _data_location(),
            "code": _git_state(),
        }
        self.text: list[str] = []
        self.items: list[dict] = []
        self._header()

    # ---- writing --------------------------------------------------------------------------
    def _out(self, s: str = "") -> None:
        print(s, flush=True)
        self.text.append(s)

    def _header(self) -> None:
        self._out("=" * WIDTH)
        self._out(f"{self.script}   {self.time:%Y-%m-%d %H:%M:%S}")
        for k, v in self.info.items():
            self._out(f"{k:<7}{v}")
        if self.args:
            self._out(f"{'args':<7}" + "  ".join(f"{k}={v}" for k, v in self.args.items()))
        self._out("=" * WIDTH)

    def section(self, title: str) -> None:
        self._out()
        self._out(f"--- {title} " + "-" * max(3, WIDTH - len(title) - 5))
        self.items.append({"type": "section", "title": title})

    def line(self, text: str = "") -> None:
        self._out(text)
        self.items.append({"type": "line", "text": text})

    def value(self, key: str, val) -> None:
        """Store a named result in the .json without printing it."""
        self.items.append({"type": "value", "key": key, "value": _plain(val)})

    def table(self, columns: list[str], rows: list[list]) -> None:
        self._print_table(columns, rows)
        self.items.append({"type": "table", "columns": columns, "rows": _plain(rows)})

    def _print_table(self, columns: list[str], rows: list[list]) -> None:
        cells = [[_cell(x) for x in r] for r in rows]
        widths = [max([len(c)] + [len(r[i]) for r in cells]) for i, c in enumerate(columns)]

        def fmt(r):  # first column left-aligned (labels), the rest right-aligned (numbers)
            return "  ".join(r[0].ljust(widths[0]) if i == 0 else r[i].rjust(widths[i]) for i in range(len(r)))

        self._out(fmt(columns))
        for r in cells:
            self._out(fmt(r))

    def scores(self, rows: list[tuple[str, dict]], label: str = "") -> None:
        """Standard score table: label, n, r, 95% CI, RMSE, then whichever of
        % of ceiling / reference r / difference CI / p the score dicts contain."""
        has = lambda key: any(key in s for _, s in rows)     # noqa: E731
        columns = [label, "n", "r", "95% CI", "RMSE"]
        columns += ["% ceiling"] * has("pct_ceiling") + ["ref r", "r - ref r (95% CI)", "p"] * has("r_ref")
        table = []
        for name, s in rows:
            r = [name, s["n"], s["r"], f"[{s['r_ci'][0]:.3f}, {s['r_ci'][1]:.3f}]", s["rmse"]]
            if has("pct_ceiling"):
                r.append(f"{s['pct_ceiling']:.0f}%" if "pct_ceiling" in s else "")
            if has("r_ref"):
                r += ([s["r_ref"], f"[{s['diff_ci'][0]:+.3f}, {s['diff_ci'][1]:+.3f}]", s["p_vs_ref"]]
                      if "r_ref" in s else ["", "", ""])
            table.append(r)
        self._print_table(columns, table)
        self.items.append({"type": "scores", "rows": [{"label": n, **_plain(s)} for n, s in rows]})

    # ---- saving and earlier runs ----------------------------------------------------------
    def save(self) -> Path:
        d = results_dir() / self.script
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{self.time:%Y-%m-%d_%H%M%S}" + (f"_{self.tag}" if self.tag else "")
        txt, js = d / f"{stem}.txt", d / f"{stem}.json"
        k = 2
        while txt.exists() or js.exists():          # never overwrite another run (runs started in the same second)
            txt, js = d / f"{stem}_run{k}.txt", d / f"{stem}_run{k}.json"; k += 1
        self._out()
        self._out(f"saved {txt.relative_to(REPO) if txt.is_relative_to(REPO) else txt} (+ .json)")
        txt.write_text("\n".join(self.text) + "\n")
        js.write_text(json.dumps({"script": self.script, "time": self.time.isoformat(timespec="seconds"),
                                  "tag": self.tag, **self.info, "args": _plain(self.args),
                                  "items": self.items}, indent=1) + "\n")
        return txt

    def previous_runs(self, **match) -> list[str]:
        """Times of earlier saved runs of this script whose args contain ``match``."""
        out = []
        for f in sorted((results_dir() / self.script).glob("*.json")):
            try:
                run = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if all(run.get("args", {}).get(k) == v for k, v in match.items()):
                out.append(run["time"])
        return out
