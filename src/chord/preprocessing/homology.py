
import hashlib
import json
from itertools import combinations

import numpy as np
import pandas as pd


def index_pair_set(pair_idx, size1=None, size2=None):
    if isinstance(pair_idx, tuple) and len(pair_idx) == 2:
        left, right = map(np.asarray, pair_idx)
        if left.ndim != 1 or right.ndim != 1 or len(left) != len(right):
            raise ValueError("Homology index vectors must be 1D and equally long.")
        pairs = np.column_stack((left, right))
    else:
        pairs = np.asarray(pair_idx)
    if pairs.size == 0:
        return set()
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
        raise ValueError("Homology pairs must contain two integer indices per row.")
    if (pairs < 0).any() or (size1 is not None and (pairs[:, 0] >= size1).any()) or (
        size2 is not None and (pairs[:, 1] >= size2).any()
    ):
        raise ValueError("Homology indices are outside the species embedding rows.")
    return set(map(tuple, pairs.tolist()))


def collect_homologous_tuples(homology_indices, species_order=None, *, species_sizes=None):
    observed = set()
    for key in homology_indices:
        if not isinstance(key, tuple) or len(key) != 2 or key[0] == key[1]:
            raise ValueError(f"Invalid homology species pair: {key!r}")
        observed.update(key)
    species = tuple(sorted(observed) if species_order is None else species_order)
    if len(species) < 2 or len(species) != len(set(species)):
        raise ValueError("Homolog tuples require at least two distinct species.")
    if set(species) != observed:
        raise ValueError("species_order must contain exactly all homology species.")
    maps = {}
    for sp1, sp2 in combinations(species, 2):
        forward, reverse = (sp1, sp2) in homology_indices, (sp2, sp1) in homology_indices
        if forward and reverse:
            raise ValueError(f"Duplicate homology species pair: {(sp1, sp2)}")
        if not forward and not reverse:
            raise ValueError(f"Missing homology mapping for species pair {(sp1, sp2)}.")
        a, b = (sp1, sp2) if forward else (sp2, sp1)
        edges = index_pair_set(
            homology_indices[(a, b)],
            None if species_sizes is None else species_sizes[a],
            None if species_sizes is None else species_sizes[b],
        )
        if reverse:
            edges = {(j, i) for i, j in edges}
        left, right = {}, {}
        for i, j in edges:
            left.setdefault(i, set()).add(j)
            right.setdefault(j, set()).add(i)
        strict = {i: next(iter(js)) for i, js in left.items()
                  if len(js) == 1 and len(right[next(iter(js))]) == 1}
        maps[(sp1, sp2)] = strict
    reference = species[0]
    tuples = []
    for ref_index in sorted(maps[(reference, species[1])]):
        row = [ref_index]
        for other in species[1:]:
            partner = maps[(reference, other)].get(ref_index)
            if partner is None:
                break
            row.append(partner)
        if len(row) == len(species) and all(
            maps[(species[i], species[j])].get(row[i]) == row[j]
            for i, j in combinations(range(len(species)), 2)
        ):
            tuples.append(tuple(row))
    if not tuples:
        raise ValueError(
            f"No complete strict one-to-one {len(species)}-species homolog tuples "
            f"exist for {list(species)} in the retained gene set. Check all pairwise "
            "homology tables and gene/HVG coverage."
        )
    return tuples


def build_homology_index_views(species_gene_names, homology_files):
    lookups = {}
    for species, names in species_gene_names.items():
        genes = list(map(str, names))
        if len(genes) != len(set(genes)) or any(not gene for gene in genes):
            raise ValueError(f"Gene IDs for {species!r} must be unique, non-empty strings.")
        lookups[species] = {gene: i for i, gene in enumerate(genes)}
    full, strict = {}, {}
    for (sp1, sp2), path in homology_files.items():
        if sp1 == sp2 or sp1 not in lookups or sp2 not in lookups:
            raise ValueError(f"Invalid homology species pair: {(sp1, sp2)}")
        if (sp2, sp1) in full:
            raise ValueError(f"Duplicate homology species pair: {(sp1, sp2)}")
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
        c1, c2 = f"{sp1}_gene", f"{sp2}_gene"
        if not {c1, c2}.issubset(table.columns):
            raise ValueError(f"Homology CSV must contain {c1!r} and {c2!r}: {path}")
        if (table[[c1, c2]] == "").any().any():
            raise ValueError(f"Homology CSV contains empty gene IDs: {path}")
        distinct = table[[c1, c2]].drop_duplicates()
        count1, count2 = distinct[c1].value_counts(), distinct[c2].value_counts()
        pairs, unique_pairs = [], set()
        for g1, g2 in zip(table[c1], table[c2]):
            if g1 in lookups[sp1] and g2 in lookups[sp2]:
                edge = (lookups[sp1][g1], lookups[sp2][g2])
                pairs.append(edge)
                if count1[g1] == 1 and count2[g2] == 1:
                    unique_pairs.add(edge)
        full[(sp1, sp2)] = ([i for i, _ in pairs], [j for _, j in pairs])
        strict[(sp1, sp2)] = ([i for i, _ in sorted(unique_pairs)],
                            [j for _, j in sorted(unique_pairs)])
    return full, strict


def homology_tuple_id(species, genes):
    text = json.dumps(sorted(zip(species, genes)), ensure_ascii=False)
    return "tuple_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
