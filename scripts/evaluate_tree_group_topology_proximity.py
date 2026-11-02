from __future__ import annotations

import argparse
import csv
import itertools
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import cophenet, is_valid_linkage
from scipy.spatial.distance import squareform

CATEGORIES = (
    "Matching labels",
    "Distinct labels, same group",
    "Different groups",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", type=Path, required=True,
                        help="CSV with 'Group' and 'Cell-type tree label' columns")
    parser.add_argument("--tree", nargs=3, action="append", required=True,
                        metavar=("METHOD", "LINKAGE_TSV", "LEAVES_TSV"),
                        help="Repeat for each method, e.g. CHORD then SATURN")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip PNG chart generation (matplotlib not required)")
    return parser.parse_args()


def csv_read(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def csv_write(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_label(label: str) -> str:
    label = label.strip()
    # Original project's occasional zebrafish_/fish_ naming discrepancy.
    if label.startswith("zebrafish_"):
        return "fish_" + label[len("zebrafish_"):]
    return label


def parse_label(label: str) -> tuple[str, str]:
    if "_" not in label:
        raise ValueError(f"Invalid cell label {label!r}: expected species_subtype")
    species, subtype = label.split("_", maxsplit=1)
    if not species or not subtype:
        raise ValueError(f"Invalid cell label {label!r}: empty species or subtype")
    return species, subtype


def load_membership(path: Path) -> tuple[OrderedDict[str, list[str]], dict[str, str]]:
    rows = csv_read(path)
    if not rows:
        raise ValueError(f"Empty group CSV: {path}")
    required = {"Group", "Cell-type tree label"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Group CSV needs {sorted(required)} (other columns allowed)")
    groups: OrderedDict[str, list[str]] = OrderedDict()
    membership: dict[str, str] = {}
    for i, row in enumerate(rows, start=2):
        group = row["Group"].strip()
        label = canonical_label(row["Cell-type tree label"])
        if not group or not label:
            raise ValueError(f"Blank Group/Cell-type tree label at CSV line {i}")
        parse_label(label)
        if label in membership:
            raise ValueError(
                f"Label {label!r} occurs in multiple rows/groups "
                f"({membership[label]!r}, {group!r}) at line {i}. "
                "These three proximity categories require NON-OVERLAPPING "
                "top-level groups; provide one Group per leaf."
            )
        groups.setdefault(group, []).append(label)
        membership[label] = group
    return groups, membership


def load_tree(name: str, linkage_path: Path, leaves_path: Path) -> dict:
    linkrows = csv_read(linkage_path, "\t")
    leafrows = csv_read(leaves_path, "\t")
    if not linkrows or not leafrows:
        raise ValueError(f"{name}: empty linkage or leaves TSV")
    try:
        leaf_indices = [int(r["leaf_index"]) for r in leafrows]
        labels = [canonical_label(r["label"]) for r in leafrows]
        z = np.asarray([[float(r[c]) for c in ("left", "right", "distance", "leaf_count")]
                        for r in linkrows], dtype=float)
    except KeyError as exc:
        raise ValueError(f"{name}: missing linkage/leaves column {exc}") from exc
    if leaf_indices != list(range(len(labels))):
        raise ValueError(f"{name}: leaf_index must be consecutive, zero based, and ordered")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name}: duplicate leaf labels after normalization")
    for label in labels:
        parse_label(label)
    if z.shape != (len(labels)-1, 4):
        raise ValueError(f"{name}: expected {(len(labels)-1, 4)} linkage array, got {z.shape}")
    is_valid_linkage(z, throw=True)
    return {"name": name, "labels": labels, "index": {x: i for i, x in enumerate(labels)},
            "D": squareform(cophenet(z))}


def percentile(value: float, sorted_bg: np.ndarray) -> float:
    low = np.searchsorted(sorted_bg, value, side="left")
    high = np.searchsorted(sorted_bg, value, side="right")
    return float((low + 0.5 * (high - low)) / len(sorted_bg))


def classify(a: str, b: str, membership: dict[str, str]) -> str | None:
    sa, ta = parse_label(a)
    sb, tb = parse_label(b)
    if sa == sb:
        return None   # Cross-species comparisons only.
    if ta == tb:
        return CATEGORIES[0]  # Matching name, including unclassified labels.
    ga, gb = membership.get(a), membership.get(b)
    if ga is None or gb is None:
        return None  # Do not treat unclassified types as a fourth class.
    return CATEGORIES[1] if ga == gb else CATEGORIES[2]


def safe_mean(values: list[float]) -> float | str:
    return float(np.mean(values)) if values else ""


def safe_median(values: list[float]) -> float | str:
    return float(np.median(values)) if values else ""


def plot_category_percentiles(summary_rows: list[dict],
                              trees: list[dict], outdir: Path) -> Path:
    """Plot the pooled mean distance percentile with exact pair-category labels."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Works on headless servers / HPC clusters.
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib or run with --no-plot") from exc

    methods = [tree["name"] for tree in trees]
    x = np.arange(len(CATEGORIES), dtype=float)
    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    bar_width = 0.75 / len(methods)
    for i, name in enumerate(methods):
        y = [float(row[f"{name}_tree_distance_mean_percentile_pct"])
             if row[f"{name}_tree_distance_mean_percentile_pct"] != "" else float("nan")
             for row in summary_rows]
        position = x + (i - (len(methods) - 1) / 2.0) * bar_width
        bars = ax.bar(position, y, width=bar_width, label=name)
        labels = [f"{v:.2f}%" if np.isfinite(v) else "n/a" for v in y]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=10)

    ax.set_xticks(x, CATEGORIES)
    ax.set_ylabel("Mean cophenetic distance percentile (%) (lower = closer)")
    ax.set_xlabel("Cross-species pair category")
    ax.set_title("Ontology-guided relative proximity")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    output = outdir / "ontology_proximity_comparison.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def evaluate(groups: OrderedDict[str, list[str]], membership: dict[str, str],
             trees: list[dict], outdir: Path, make_plot: bool = True) -> None:
    if len({t["name"] for t in trees}) != len(trees):
        raise ValueError("--tree method names must be unique")
    expected = set(trees[0]["labels"])
    for tree in trees[1:]:
        if set(tree["labels"]) != expected:
            raise ValueError(f"Trees {trees[0]['name']} and {tree['name']} have different leaves")
    missing = set(membership) - expected
    if missing:
        raise ValueError(f"Group file references leaves missing from trees: {sorted(missing)}")
    outdir.mkdir(parents=True, exist_ok=True)

    species_pairs = sorted({tuple(sorted((parse_label(a)[0], parse_label(b)[0])))
                            for a, b in itertools.combinations(trees[0]["labels"], 2)
                            if parse_label(a)[0] != parse_label(b)[0]})
    # Use a common, deterministic pair list for every method, even if leaf order differs.
    ordered_labels = sorted(expected)
    all_cross = []
    classified = []
    excluded_missing = 0
    for a, b in itertools.combinations(ordered_labels, 2):
        sa, _ = parse_label(a)
        sb, _ = parse_label(b)
        if sa == sb:
            continue
        sp = tuple(sorted((sa, sb)))
        all_cross.append((a, b, sp))
        cat = classify(a, b, membership)
        if cat is None:
            excluded_missing += 1
        else:
            classified.append((a, b, sp, cat))

    if not all_cross:
        raise ValueError("At least two species are needed for cross-species proximity")

    detail_rows: list[dict] = []
    by_pair_rows: list[dict] = []
    pooled_stats: dict[tuple[str, str], dict] = {}
    for tree in trees:
        indices, D = tree["index"], tree["D"]
        backgrounds = defaultdict(list)
        for a, b, sp in all_cross:
            backgrounds[sp].append(float(D[indices[a], indices[b]]))
        backgrounds = {sp: np.sort(np.asarray(vals, dtype=float))
                       for sp, vals in backgrounds.items()}

        raw = defaultdict(list)
        ranks = defaultdict(list)
        for a, b, sp, cat in classified:
            d = float(D[indices[a], indices[b]])
            rank = percentile(d, backgrounds[sp])
            key = (sp, cat)
            raw[key].append(d)
            ranks[key].append(rank)
            raw[("pooled", cat)].append(d)
            ranks[("pooled", cat)].append(rank)
            detail_rows.append({
                "Method": tree["name"], "Species pair": "-".join(sp),
                "Pair category": cat,
                "Leaf A": a, "Leaf B": b,
                "Group A": membership.get(a, "Unclassified"),
                "Group B": membership.get(b, "Unclassified"),
                "Cophenetic distance": d,
                "Distance percentile": rank,
                "Distance percentile (%)": rank * 100,
            })

        for cat in CATEGORIES:
            key = ("pooled", cat)
            pooled_stats[(tree["name"], cat)] = {
                "nPairs": len(ranks[key]),
                "mean_raw": safe_mean(raw[key]),
                "mean_percentile": safe_mean(ranks[key]),
                "median_percentile": safe_median(ranks[key]),
            }
            for sp in species_pairs:
                k = (sp, cat)
                by_pair_rows.append({
                    "Method": tree["name"], "Species pair": "-".join(sp),
                    "Pair category": cat, "nPairs": len(ranks[k]),
                    "Mean cophenetic distance": safe_mean(raw[k]),
                    "Mean distance percentile": safe_mean(ranks[k]),
                    "Mean distance percentile (%)": (100 * np.mean(ranks[k]) if ranks[k] else ""),
                    "Median distance percentile (%)": (100 * np.median(ranks[k]) if ranks[k] else ""),
                    "Background nPairs": len(backgrounds[sp]),
                })

    summary_rows = []
    for cat in CATEGORIES:
        row = {"Group": cat, "nPairs": pooled_stats[(trees[0]["name"], cat)]["nPairs"]}
        for t in trees:
            m = t["name"]
            p = pooled_stats[(m, cat)]
            row[f"{m}_tree_cophenetic_mean"] = p["mean_raw"]
            row[f"{m}_tree_distance_mean_percentile"] = p["mean_percentile"]
            row[f"{m}_tree_distance_mean_percentile_pct"] = (
                100 * p["mean_percentile"] if p["mean_percentile"] != "" else "")
            row[f"{m}_tree_distance_median_percentile_pct"] = (
                100 * p["median_percentile"] if p["median_percentile"] != "" else "")
        summary_rows.append(row)

    summary_cols = ["Group", "nPairs"]
    for t in trees:
        m = t["name"]
        summary_cols.extend([
            f"{m}_tree_cophenetic_mean",
            f"{m}_tree_distance_mean_percentile",
            f"{m}_tree_distance_mean_percentile_pct",
            f"{m}_tree_distance_median_percentile_pct",
        ])
    csv_write(outdir / "group_tree_summary.csv", summary_rows, summary_cols)
    csv_write(outdir / "group_tree_by_species_pair.csv", by_pair_rows, [
        "Method", "Species pair", "Pair category", "nPairs", "Mean cophenetic distance",
        "Mean distance percentile", "Mean distance percentile (%)",
        "Median distance percentile (%)", "Background nPairs",
    ])
    csv_write(outdir / "group_tree_pairwise_distances.csv", detail_rows, [
        "Method", "Species pair", "Pair category", "Leaf A", "Leaf B", "Group A", "Group B",
        "Cophenetic distance", "Distance percentile", "Distance percentile (%)",
    ])

    print(f"Validated {len(trees)} trees with {len(expected)} common leaves "
          f"and {len(groups)} nonoverlapping reference groups")
    print(f"Cross-species background: {len(all_cross)} pairs "
          f"(including {excluded_missing} unclassified/unmatched-subtype pairs "
          "excluded from the 3 category scores)")
    for cat in CATEGORIES:
        vals = [f"{t['name']}: " +
                (f"{pooled_stats[(t['name'], cat)]['mean_percentile']*100:.2f}% "
                 f"(raw={pooled_stats[(t['name'], cat)]['mean_raw']:.6f})"
                 if pooled_stats[(t['name'], cat)]['nPairs'] else "NA")
                for t in trees]
        print(f"{cat} | nPairs={pooled_stats[(trees[0]['name'], cat)]['nPairs']} | "
              + " | ".join(vals))
    print("Saved: group_tree_summary.csv, group_tree_by_species_pair.csv, "
          "group_tree_pairwise_distances.csv")
    if make_plot:
        output = plot_category_percentiles(summary_rows, trees, outdir)
        print(f"Saved: {output}")


def main() -> None:
    args = parse_args()
    groups, membership = load_membership(args.groups)
    trees = [load_tree(name, Path(link), Path(leaves))
             for name, link, leaves in args.tree]
    evaluate(groups, membership, trees, args.outdir, make_plot=not args.no_plot)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
