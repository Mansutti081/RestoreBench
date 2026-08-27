# ABOUTME: CLI wrapper for rendering FULL pandapower benchmark states into Scenario Cards.
# ABOUTME: Re-exports the backend renderer so dataset scripts keep their historical import path.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandapower as pp


from restorebench.environment.card_render import (
    _active,
    _fmt_bool,
    _fmt_float,
    _fmt_int,
    _gen_rows,
    _neighbors,
    _section_loads,
    _section_setpoints,
    _section_topology,
    _shunt_rows,
    _trafo_rows,
    render_scenario_card,
)


ROOT = Path(__file__).resolve().parents[2]
__all__ = [
    "_active",
    "_fmt_bool",
    "_fmt_float",
    "_fmt_int",
    "_gen_rows",
    "_neighbors",
    "_section_loads",
    "_section_setpoints",
    "_section_topology",
    "_shunt_rows",
    "_trafo_rows",
    "render_all",
    "render_file",
    "render_scenario_card",
]

FULL_DIR = ROOT / "dataset/ieee118/full"
LLM_DIR = ROOT / "dataset/ieee118/llm"


def render_file(in_path: str | Path, out_path: str | Path) -> None:
    """Load one FULL pandapower JSON, render its Card, and write Markdown."""
    net = pp.from_json(str(in_path))
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_scenario_card(net), encoding="utf-8")


def render_all(dataset_dir: str | Path | None = None) -> int:
    """Render every FULL scenario JSON in a dataset directory into matching LLM Cards."""
    if dataset_dir is None:
        full_dir = FULL_DIR
        llm_dir = LLM_DIR
    else:
        root = Path(dataset_dir)
        full_dir = root / "full"
        llm_dir = root / "llm"
    if not full_dir.exists():
        print(f"No full/ directory found at {full_dir}")
        return 0
    files = sorted(full_dir.glob("*.json"))
    for in_path in files:
        out_path = llm_dir / f"{in_path.stem}.md"
        render_file(in_path, out_path)
        print(f"  {in_path.name}  ->  llm/{out_path.name}")
    print(f"Rendered {len(files)} file(s) into {llm_dir}")
    return len(files)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset-dir", type=Path)
    args = parser.parse_args(argv)
    if args.all:
        if args.dataset_dir is None or args.paths:
            parser.error("--all requires --dataset-dir and no positional paths")
    elif len(args.paths) != 2 or args.dataset_dir is not None:
        parser.error("provide input and output paths, or --all --dataset-dir")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.all:
        render_all(args.dataset_dir)
        return
    render_file(args.paths[0], args.paths[1])
    print(f"Rendered {args.paths[0]}  ->  {args.paths[1]}")


if __name__ == "__main__":
    main(sys.argv[1:])
