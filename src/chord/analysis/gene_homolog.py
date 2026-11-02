import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
from itertools import combinations, permutations
from pathlib import Path
from scipy.sparse import issparse
from ..utils.plot import pretty_boxplot
from ..preprocessing.homology import (
    collect_homologous_tuples,
    homology_tuple_id,
    index_pair_set as _index_pairs,
)


def cosine_similarity_paired(X1, X2, eps=1e-12):
    numerator = (X1 * X2).sum(axis=1)
    denominator = np.linalg.norm(X1, axis=1) * np.linalg.norm(X2, axis=1) + eps
    return numerator / denominator


def split_embedding_by_species(adata_gene, species_key="species"):
    X = adata_gene.X
    species_arr = adata_gene.obs[species_key].values
    species_to_global = {
        sp: np.where(species_arr == sp)[0]
        for sp in np.unique(species_arr)
    }
    return X, species_arr, species_to_global


def _bh_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    order = valid[np.argsort(values[valid])]
    if len(order):
        ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
        adjusted[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return adjusted


def evaluate_homolog_embeddings(
    adata_gene,
    homology_indices,
    *,
    species_key: str = "species",
    random_seed: int = 42,
    make_plot: bool = False,
    gene_key: str = "gene_id",
    pair_selection: str = "auto",
    known_homology_indices=None,
    n_randomizations: int = 100,
    n_permutations: int = 9999,
    max_shuffle_attempts: int = 10000,
    plot_path="box.pdf",
    return_details: bool = False,
):
    if pair_selection not in {"auto", "all", "withheld"}:
        raise ValueError("pair_selection must be auto, all, or withheld.")
    for name, value in (("n_randomizations", n_randomizations),
                        ("n_permutations", n_permutations),
                        ("max_shuffle_attempts", max_shuffle_attempts)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if isinstance(random_seed, bool) or not isinstance(random_seed, (int, np.integer)) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer.")
    if not homology_indices:
        raise ValueError("homology_indices must contain at least one species pair.")

    withheld = adata_gene.uns.get("withheld_homology_pairs")
    withholding_enabled = bool(adata_gene.uns.get("homology_anchor_withholding_enabled", False))
    selection = pair_selection
    if selection == "auto":
        selection = "withheld" if withholding_enabled or withheld is not None else "all"
    if selection == "withheld":
        required = {"species1", "species2", "gene1", "gene2"}
        if not isinstance(withheld, pd.DataFrame) or not required.issubset(withheld.columns):
            raise ValueError("Withheld evaluation requires uns['withheld_homology_pairs'] with species/gene columns.")
        if gene_key not in adata_gene.obs:
            raise ValueError(f"Withheld evaluation requires obs[{gene_key!r}].")

    emb, _, species_to_global = split_embedding_by_species(
        adata_gene, species_key=species_key
    )
    emb = emb.toarray() if issparse(emb) else np.asarray(emb)
    if emb.ndim != 2 or not np.isfinite(emb).all():
        raise ValueError("Gene embeddings must be a finite 2D matrix.")
    names = {}
    for sp, positions in species_to_global.items():
        names[sp] = (adata_gene.obs.iloc[positions][gene_key].astype(str).to_numpy()
                     if gene_key in adata_gene.obs else np.asarray([str(j) for j in range(len(positions))]))
        if len(set(names[sp])) != len(names[sp]):
            raise ValueError(f"Gene IDs must be unique within species {sp!r}.")

    def read_pairs(mapping, sp1, sp2):
        pairs = _index_pairs(mapping.get((sp1, sp2), []), len(names[sp1]), len(names[sp2]))
        reverse = _index_pairs(mapping.get((sp2, sp1), []), len(names[sp2]), len(names[sp1]))
        return pairs | {(b, a) for a, b in reverse}

    rows, pair_frames, control_rows, null_rows, plot_data = [], [], [], [], []
    for sp1, sp2 in homology_indices:
        if sp1 == sp2 or sp1 not in names or sp2 not in names:
            raise ValueError(f"Invalid embedding species pair: {(sp1, sp2)}")
        full = read_pairs(homology_indices, sp1, sp2)
        known = full | read_pairs(known_homology_indices or {}, sp1, sp2)
        selected = full
        if selection == "withheld":
            name1 = {name: j for j, name in enumerate(names[sp1])}
            name2 = {name: j for j, name in enumerate(names[sp2])}
            selected = set()
            for item in withheld.itertuples(index=False):
                if (str(item.species1), str(item.species2)) == (sp1, sp2):
                    g1, g2 = str(item.gene1), str(item.gene2)
                elif (str(item.species2), str(item.species1)) == (sp1, sp2):
                    g1, g2 = str(item.gene2), str(item.gene1)
                else:
                    continue
                if g1 not in name1 or g2 not in name2:
                    raise ValueError(f"Withheld pair {(sp1, g1, sp2, g2)} is missing from embeddings.")
                pair = (name1[g1], name2[g2])
                if pair not in known:
                    raise ValueError(f"Withheld pair {(sp1, g1, sp2, g2)} is missing from known homology.")
                selected.add(pair)
        left_counts, right_counts = {}, {}
        for a, b in known:
            left_counts[a] = left_counts.get(a, 0) + 1
            right_counts[b] = right_counts.get(b, 0) + 1
        strict = sorted((a, b) for a, b in selected if left_counts[a] == right_counts[b] == 1)
        if selection == "withheld" and len(strict) != len(selected):
            raise ValueError("Saved withheld pairs are no longer strict one-to-one in known homology.")
        loc1, loc2 = np.asarray(strict, dtype=int).reshape(-1, 2).T
        n = len(strict)
        row = dict(species1=sp1, species2=sp2, pair_selection=selection,
                   n_input_homolog=len(selected), n_homolog=n,
                   n_excluded_non_one_to_one=len(selected) - n,
                   status="ok", median_homolog=np.nan, median_random=np.nan,
                   mean_homolog=np.nan, mean_random=np.nan, mean_difference=np.nan,
                   p=np.nan, p_method="none", n_randomizations=0, n_permutations=0)
        idx1_global = species_to_global[sp1][loc1]
        idx2_global = species_to_global[sp2][loc2]
        X1, X2 = emb[idx1_global].astype(float), emb[idx2_global].astype(float)
        norm1, norm2 = np.linalg.norm(X1, axis=1), np.linalg.norm(X2, axis=1)

        def scores(order):
            return np.einsum("ij,ij->i", X1, X2[order]) / (norm1 * norm2[order] + 1e-12)

        cos_hom = scores(np.arange(n))
        if n:
            row.update(median_homolog=float(np.median(cos_hom)), mean_homolog=float(cos_hom.mean()))

        def record_pairs(group, round_id, order, values):
            if return_details:
                pair_frames.append(pd.DataFrame(dict(
                    species1=sp1, species2=sp2, pair_selection=selection,
                    group=group, round=round_id, gene1=names[sp1][loc1],
                    gene2=names[sp2][loc2[order]], similarity=values)))

        record_pairs("Ortholog", -1, np.arange(n), cos_hom)
        if n < 2:
            row["status"] = "skipped_no_pairs" if n == 0 else "skipped_insufficient_pairs"
            rows.append(row)
            plot_data.append((row, cos_hom, np.array([])))
            continue

        key = hashlib.sha256(f"{sp1}\0{sp2}".encode()).digest()
        streams = np.random.SeedSequence([int(random_seed), int.from_bytes(key[:8], "little")]).spawn(2)
        control_rng, test_rng = (np.random.default_rng(s) for s in streams)
        control_means, first_random = [], None
        for round_id in range(n_randomizations):
            for attempt in range(max_shuffle_attempts):
                order = control_rng.permutation(n)
                if all((int(a), int(b)) not in known for a, b in zip(loc1, loc2[order])):
                    break
            else:
                raise RuntimeError(f"No valid shuffled pairing for {(sp1, sp2)} after {max_shuffle_attempts} attempts.")
            values = scores(order)
            if first_random is None:
                first_random = values.copy()
            mean = float(values.mean())
            control_means.append(mean)
            record_pairs("Shuffled", round_id, order, values)
            if return_details:
                control_rows.append(dict(species1=sp1, species2=sp2, round=round_id,
                                         mean_similarity=mean, mean_difference=row["mean_homolog"] - mean))

        pairing_count = 1
        for size in range(2, n + 1):
            pairing_count *= size
            if pairing_count > n_permutations:
                break
        exact = pairing_count <= n_permutations
        orders = permutations(range(n)) if exact else (test_rng.permutation(n) for _ in range(n_permutations))
        exceedances, draws = 0, 0
        for draw, order in enumerate(orders):
            mean = float(scores(np.asarray(order)).mean())
            exceedances += int(mean >= row["mean_homolog"] - 1e-12)
            draws += 1
            if return_details:
                null_rows.append(dict(species1=sp1, species2=sp2, draw=draw, mean_similarity=mean))
        row.update(median_random=float(np.median(first_random)), mean_random=float(np.mean(control_means)),
                   mean_difference=row["mean_homolog"] - float(np.mean(control_means)),
                   p=exceedances / draws if exact else (exceedances + 1) / (draws + 1),
                   p_method="exact_pairing_permutation" if exact else "monte_carlo_pairing_permutation",
                   n_randomizations=n_randomizations, n_permutations=draws)
        rows.append(row)
        plot_data.append((row, cos_hom, first_random))

    summary = pd.DataFrame(rows)
    summary["p_adjusted_bh"] = _bh_adjust(summary["p"])
    if make_plot:
        destination = Path(plot_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, len(plot_data), figsize=(4 * len(plot_data), 4), squeeze=False)
        for ax, (row, true_values, random_values) in zip(axes.ravel(), plot_data):
            if row["status"] != "ok":
                ax.text(0.5, 0.5, row["status"], ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{row['species1']} / {row['species2']}")
                continue
            pretty_boxplot(
                ax, colors=["#F3756D", "#1FBDC3"],
                data=[true_values, random_values], labels=["Ortholog", "Shuffled"],
                title=f"{row['species1']} / {row['species2']} ({selection})",
                pval=row["p"], ylim=(-1.05, 1.05),
            )
            ax.set_ylabel("Cosine similarity")
        fig.tight_layout()
        try:
            fig.savefig(destination)
        finally:
            plt.close(fig)
    if not return_details:
        return summary
    pair_columns = ["species1", "species2", "pair_selection", "group", "round", "gene1", "gene2", "similarity"]
    details = {
        "pair_scores": pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame(columns=pair_columns),
        "randomization_scores": pd.DataFrame(control_rows, columns=["species1", "species2", "round", "mean_similarity", "mean_difference"]),
        "permutation_scores": pd.DataFrame(null_rows, columns=["species1", "species2", "draw", "mean_similarity"]),
    }
    return summary, details


def _tuple_species(table, species_order):
    species = tuple(species_order if species_order is not None else
                    table.attrs.get("species_order", [c[:-5] for c in table.columns if c.endswith("_gene")]))
    if len(species) < 2 or len(species) != len(set(species)):
        raise ValueError("Tuple similarities require at least two distinct species.")
    columns = [f"{a}_{b}_sim" for a, b in combinations(species, 2)]
    if len(columns) != len(set(columns)):
        raise ValueError("Species names produce ambiguous similarity column names.")
    if not set(columns).issubset(table.columns):
        raise ValueError(f"Tuple table must contain similarity columns {columns}.")
    if table.empty or not np.isfinite(table[columns].to_numpy(dtype=float)).all():
        raise ValueError("Tuple similarities must be non-empty and finite.")
    if "tuple_id" not in table or table["tuple_id"].duplicated().any():
        raise ValueError("Tuple similarities must contain unique tuple_id values.")
    return species, columns


def compute_homolog_tuple_similarity_table(
    adata_gene, homology_indices, *, species_key="species", gene_key="gene_id",
    species_order=None, tuple_selection="auto",
):
    if tuple_selection not in {"auto", "all", "withheld"}:
        raise ValueError("tuple_selection must be auto, all, or withheld.")
    if not {species_key, gene_key}.issubset(adata_gene.obs.columns):
        raise ValueError(f"Gene embeddings require obs[{species_key!r}] and obs[{gene_key!r}].")
    obs_species = adata_gene.obs[species_key].astype(str).to_numpy()
    genes = adata_gene.obs[gene_key].astype(str).to_numpy()
    species = tuple(sorted(set(obs_species)) if species_order is None else species_order)
    if set(species) != set(obs_species) or len(species) != len(set(species)):
        raise ValueError("species_order must contain exactly all embedding species.")
    positions = {sp: np.flatnonzero(obs_species == sp) for sp in species}
    names = {sp: genes[idx] for sp, idx in positions.items()}
    if any(len(set(values)) != len(values) for values in names.values()):
        raise ValueError("Gene IDs must be unique within each species.")
    lookups = {sp: {gene: i for i, gene in enumerate(values)} for sp, values in names.items()}
    complete = collect_homologous_tuples(homology_indices, species,
                                       species_sizes={sp: len(values) for sp, values in names.items()})
    saved = adata_gene.uns.get("withheld_homology_tuples")
    enabled = bool(adata_gene.uns.get("homology_anchor_withholding_enabled", False))
    if tuple_selection == "auto":
        tuple_selection = "withheld" if enabled or saved is not None else "all"
    if tuple_selection == "withheld":
        if adata_gene.uns.get("homology_anchor_withholding_mode") != "shared_tuples" or not isinstance(saved, pd.DataFrame):
            raise ValueError("Withheld tuple evaluation requires mode='shared_tuples' and a saved withheld_homology_tuples table; legacy pairwise masks are insufficient.")
        if set(map(str, adata_gene.uns.get("homology_anchor_withholding_species", []))) != set(species):
            raise ValueError("Saved tuple species do not match the embedding species.")
        required = ["tuple_id"] + [f"{sp}_gene" for sp in species]
        if not set(required).issubset(saved.columns) or saved.empty:
            raise ValueError("No valid saved withheld tuples; use a positive withholding fraction and matching outputs.")
        records = saved[required].astype(str).to_dict("records")
        if saved["tuple_id"].duplicated().any() or any(saved[f"{sp}_gene"].duplicated().any() for sp in species):
            raise ValueError("Saved withheld tuples contain duplicate tuple IDs or genes.")
        tuples = []
        complete_set = set(complete)
        for record in records:
            try:
                row = tuple(lookups[sp][record[f"{sp}_gene"]] for sp in species)
            except KeyError as exc:
                raise ValueError(f"A saved withheld tuple gene is missing from current embeddings: {exc}") from exc
            if row not in complete_set:
                raise ValueError("A saved withheld tuple is not a complete strict one-to-one tuple in the supplied homology mappings.")
            tuples.append(row)
        pairs = adata_gene.uns.get("withheld_homology_pairs")
        required_pairs = {"tuple_id", "species1", "species2", "gene1", "gene2"}
        if not isinstance(pairs, pd.DataFrame) or not required_pairs.issubset(pairs.columns):
            raise ValueError("Saved synchronized tuples require their complete withheld_homology_pairs manifest.")
        order = {sp: i for i, sp in enumerate(species)}
        expected = {(r["tuple_id"], a, r[f"{a}_gene"], b, r[f"{b}_gene"])
                    for r in records for a, b in combinations(species, 2)}
        actual = []
        for row in pairs.itertuples(index=False):
            a, b, ga, gb = str(row.species1), str(row.species2), str(row.gene1), str(row.gene2)
            if a not in order or b not in order or a == b:
                raise ValueError("Invalid species in the withheld pair manifest.")
            if order[a] > order[b]:
                a, b, ga, gb = b, a, gb, ga
            actual.append((str(row.tuple_id), a, ga, b, gb))
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Withheld tuple and pair manifests disagree; all n*(n-1)/2 edges must have been withheld together.")
    else:
        tuples = sorted(complete, key=lambda row: tuple(names[sp][i] for sp, i in zip(species, row)))
        records = []
        for row in tuples:
            row_genes = [names[sp][i] for sp, i in zip(species, row)]
            records.append({"tuple_id": homology_tuple_id(species, row_genes),
                            **{f"{sp}_gene": gene for sp, gene in zip(species, row_genes)}})
    X = adata_gene.X.toarray() if issparse(adata_gene.X) else np.asarray(adata_gene.X)
    if X.ndim != 2 or len(X) != len(obs_species):
        raise ValueError("Gene embedding matrix does not match obs rows.")
    vectors = {sp: X[positions[sp][[row[i] for row in tuples]]].astype(float)
               for i, sp in enumerate(species)}
    if any(not np.isfinite(v).all() or (np.linalg.norm(v, axis=1) == 0).any() for v in vectors.values()):
        raise ValueError("Selected gene embeddings must be finite and have nonzero norms.")
    table = pd.DataFrame(records)
    for a, b in combinations(species, 2):
        table[f"{a}_{b}_sim"] = cosine_similarity_paired(vectors[a], vectors[b])
    table.attrs.update({"species_order": list(species), "tuple_selection": tuple_selection})
    _tuple_species(table, species)
    return table


def plot_homolog_tuple_similarities(
    similarity_table, *, species_order=None, plot_path="homolog_tuple_boxplot.pdf",
    png_path=None, title=None,
):
    species, columns = _tuple_species(similarity_table, species_order)
    labels = [f"{a}–{b}" for a, b in combinations(species, 2)]
    path = Path(plot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if png_path is not None:
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, 1.05 * len(columns)), 4.5))
    try:
        colors = [plt.get_cmap("tab20")(i % 20) for i in range(len(columns))]
        scope = similarity_table.attrs.get("tuple_selection", "selected")
        pretty_boxplot(
            ax, [similarity_table[c].to_numpy() for c in columns], labels,
            title or f"{scope.capitalize()} ortholog tuples (n = {len(similarity_table)})",
            None, colors=colors, ylim=(-1.05, 1.05),
        )
        ax.set_ylabel("Gene embedding cosine similarity")
        ax.tick_params(axis="x", labelrotation=30 if len(columns) <= 6 else 50)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        if png_path is not None:
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)


def evaluate_homolog_tuple_similarities(
    similarity_table, *, species_order=None, comparisons=None, alternative="two-sided",
    random_seed=42, n_permutations=9999, return_details=False,
):
    species, columns = _tuple_species(similarity_table, species_order)
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be two-sided, greater, or less.")
    if isinstance(random_seed, bool) or not isinstance(random_seed, (int, np.integer)) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer.")
    if isinstance(n_permutations, bool) or not isinstance(n_permutations, (int, np.integer)) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer.")
    pair_names = list(combinations(species, 2))
    canonical = {frozenset(pair): pair for pair in pair_names}
    requested = list(combinations(pair_names, 2)) if comparisons is None else list(comparisons)
    tests, seen = [], set()
    for pair1, pair2 in requested:
        if len(pair1) != 2 or len(pair2) != 2 or frozenset(pair1) not in canonical or frozenset(pair2) not in canonical:
            raise ValueError("Comparisons must contain two valid species pairs.")
        a, b = canonical[frozenset(pair1)], canonical[frozenset(pair2)]
        key = frozenset((a, b))
        if a == b or key in seen:
            raise ValueError("Comparisons must be distinct and cannot repeat a test.")
        seen.add(key)
        tests.append((a, b))
    rows, differences, null_rows = [], [], []
    for pair1, pair2 in tests:
        c1, c2 = [f"{a}_{b}_sim" for a, b in (pair1, pair2)]
        score1, score2 = similarity_table[c1].to_numpy(float), similarity_table[c2].to_numpy(float)
        delta = score1 - score2
        observed = float(delta.mean())
        label = f"{pair1[0]}–{pair1[1]} vs {pair2[0]}–{pair2[1]}"
        n = len(delta)
        p, method, count = np.nan, "insufficient_tuples", 0
        if n >= 2:
            exact = 2 ** n <= n_permutations
            count = 2 ** n if exact else int(n_permutations)
            method = "exact_paired_sign_permutation" if exact else "monte_carlo_paired_sign_permutation"
            digest = hashlib.sha256(repr((pair1, pair2)).encode()).digest()
            rng = np.random.default_rng(np.random.SeedSequence([int(random_seed), int.from_bytes(digest[:8], "little")]))
            exceedances = 0
            tolerance = 1e-12 * max(1.0, abs(observed))
            for start in range(0, count, 128):
                stop = min(start + 128, count)
                if exact:
                    signs = ((np.arange(start, stop, dtype=np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float) * 2 - 1
                else:
                    signs = rng.integers(0, 2, size=(stop - start, n)) * 2 - 1
                null = (signs @ delta) / n
                if alternative == "two-sided":
                    exceedances += int((np.abs(null) >= abs(observed) - tolerance).sum())
                elif alternative == "greater":
                    exceedances += int((null >= observed - tolerance).sum())
                else:
                    exceedances += int((null <= observed + tolerance).sum())
                if return_details:
                    null_rows.extend({"comparison": label, "draw": start + i, "mean_difference": value}
                                     for i, value in enumerate(null))
            p = exceedances / count if exact else (exceedances + 1) / (count + 1)
        rows.append({"comparison": label, "pair1_species1": pair1[0], "pair1_species2": pair1[1],
                     "pair2_species1": pair2[0], "pair2_species2": pair2[1], "n_tuples": n,
                     "mean_similarity1": score1.mean(), "mean_similarity2": score2.mean(),
                     "median_similarity1": np.median(score1), "median_similarity2": np.median(score2),
                     "mean_difference": observed, "median_difference": np.median(delta),
                     "fraction_pair1_higher": (delta > 0).mean(), "alternative": alternative,
                     "status": "ok" if n >= 2 else "insufficient_tuples", "p": p,
                     "p_method": method, "n_permutations": count})
        if return_details:
            differences.extend({"tuple_id": tid, "comparison": label, "difference": value}
                               for tid, value in zip(similarity_table["tuple_id"], delta))
    fields = ["comparison", "pair1_species1", "pair1_species2", "pair2_species1", "pair2_species2", "n_tuples",
              "mean_similarity1", "mean_similarity2", "median_similarity1", "median_similarity2", "mean_difference",
              "median_difference", "fraction_pair1_higher", "alternative", "status", "p", "p_method", "n_permutations"]
    summary = pd.DataFrame(rows, columns=fields)
    summary["p_adjusted_bh"] = _bh_adjust(summary["p"])
    if not return_details:
        return summary
    return summary, {
        "paired_differences": pd.DataFrame(differences, columns=["tuple_id", "comparison", "difference"]),
        "permutation_scores": pd.DataFrame(null_rows, columns=["comparison", "draw", "mean_difference"]),
    }
