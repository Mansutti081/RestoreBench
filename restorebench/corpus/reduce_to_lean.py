# ABOUTME: Reduces a COMPLETE pandapower network JSON to a LEAN one in the SAME pandapower format.
# ABOUTME: Drops every column not controllable by the actions (keeps topology); strips PF results.

import argparse
import sys
from pathlib import Path

import pandapower as pp


# Project root = two levels up from this file (restorebench/corpus/ -> repo root).
# Used so --all mode finds the dataset folders regardless of the current working directory.
ROOT = Path(__file__).resolve().parents[2]
FULL_DIR = ROOT / "dataset/ieee118/full"
LEAN_DIR = ROOT / "dataset/ieee118/lean"

# The LEAN view contains current electrical state and declared movement/feasibility limits.
# It excludes result tables, costs, private recipes, witnesses, and curation evidence.
COLUMN_WHITELIST = {
    "bus": ["vn_kv", "in_service"],
    "line": ["from_bus", "to_bus", "in_service"],
    "trafo": [
        "hv_bus",
        "lv_bus",
        "tap_side",
        "tap_pos",
        "tap_min",
        "tap_max",
        "tap_step_percent",
        "in_service",
    ],
    "shunt": ["bus", "p_mw", "q_mvar", "step", "max_step", "in_service"],
    "gen": [
        "bus",
        "vm_pu",
        "p_mw",
        "min_p_mw",
        "max_p_mw",
        "min_q_mvar",
        "max_q_mvar",
        "in_service",
    ],
    "load": ["bus", "p_mw", "q_mvar", "in_service"],
    "ext_grid": [
        "bus",
        "vm_pu",
        "min_p_mw",
        "max_p_mw",
        "min_q_mvar",
        "max_q_mvar",
        "in_service",
    ],
}

DROP_TABLES = ["poly_cost"]


def reduce_to_lean(net):
    """
    Reduce a pandapower network IN PLACE and return it, so it can be written back with
    pp.to_json in the exact same format as the input file:
      - clear the power-flow results,
      - in each whitelisted table, drop every column that is not in COLUMN_WHITELIST,
      - empty the tables that carry only non-controllable data (DROP_TABLES).
    The result is a normal pandapower net (still pp.to_json / pp.from_json compatible) that
    simply no longer contains the non-controllable parameters.
    """
    # net.res_* tables hold a previous power-flow's output. Clear them so the LEAN state
    # describes settings only — each tool/agent re-runs the power flow itself.
    pp.reset_results(net)

    # Strip non-controllable columns from each whitelisted table.
    for table, keep in COLUMN_WHITELIST.items():
        df = getattr(net, table)              # e.g. net.line — a pandas DataFrame
        drop = [c for c in df.columns if c not in keep]
        df.drop(columns=drop, inplace=True)   # remove those columns in place

    # Empty the fully non-controllable tables (keep the table, drop all its rows).
    for table in DROP_TABLES:
        df = getattr(net, table)
        setattr(net, table, df.iloc[0:0])     # df.iloc[0:0] = same columns, zero rows

    return net


def reduce_file(in_path, out_path) -> None:
    """Load a complete network JSON, reduce it, and write the LEAN JSON in pandapower format."""
    net = pp.from_json(str(in_path))          # read the full network
    reduce_to_lean(net)                       # strip it down in place
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)   # create lean/ if missing
    pp.to_json(net, str(out_path))            # write back in the SAME pandapower format


def reduce_all(dataset_dir: str | Path | None = None) -> int:
    """Batch mode: reduce every FULL JSON under one explicit dataset root."""
    if dataset_dir is None:
        full_dir = FULL_DIR
        lean_dir = LEAN_DIR
    else:
        root = Path(dataset_dir)
        full_dir = root / "full"
        lean_dir = root / "lean"
    if not full_dir.exists():
        print(f"No full/ directory found at {full_dir}")
        return 0
    files = sorted(full_dir.glob("*.json"))
    for in_path in files:
        out_path = lean_dir / in_path.name
        reduce_file(in_path, out_path)
        print(f"  {in_path.name}  ->  lean/{in_path.name}")
    print(f"Reduced {len(files)} file(s) into {lean_dir}")
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
        reduce_all(args.dataset_dir)
        return
    reduce_file(args.paths[0], args.paths[1])
    print(f"Reduced {args.paths[0]}  ->  {args.paths[1]}")


if __name__ == "__main__":
    main(sys.argv[1:])
