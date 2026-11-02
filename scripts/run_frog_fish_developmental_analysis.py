
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_ROOT / "results" / "frog_fish"
sys.path.insert(0, str(CODE_ROOT / "src"))

from chord.analysis import run_developmental_dissimilarity_analysis


EMBEDDING_H5AD = PROJECT_ROOT / "adata_cell_embeddings.h5ad"
OUTPUT_DIR = PROJECT_ROOT / "frog_fish_developmental_dissimilarity"

MODEL_PATH = None
TRAINING_CONFIG_PATH = None
DEVICE = "auto"
INFERENCE_BATCH_SIZE = 256

SIMILARITY_MATRIX_PATH = None
SIMILARITY_MATRIX_METHOD = None

SPECIES_COL = "species"
TIME_COL = "Time"
CELL_ID_COL = "cell_id"
CELL_TYPE_COL = "cell_type"
EMBEDDING_KEY = "X"

SPECIES_METADATA = {
    "fish": {
        "h5ad": str(CODE_ROOT / "data" / "fish.h5ad"),
        "time_col": "TimeID",
        "cell_type_col": "cell_type",
        "cell_id_col": None,
    },
    "frog": {
        "h5ad": str(CODE_ROOT / "data" / "frog.h5ad"),
        "time_col": "stage",
        "cell_type_col": "cell_type",
        "cell_id_col": None,
    },
}

SPECIES_PAIR = ("frog", "fish")
ALIGNMENT_TRANSPOSE_AXES = True
ALIGNMENT_QUERY_AXIS = "rows"
DTW_OPEN_END = True
TIME_ORDERS = {
    "frog": ["Stage_8", "Stage_10", "Stage_11", "Stage_12", "Stage_13",
             "Stage_14", "Stage_16", "Stage_18", "Stage_20", "Stage_22"],
    "fish": ["4hpf", "6hpf", "8hpf", "10hpf", "14hpf", "18hpf", "24hpf"],
}
ALIGNMENT_RANGES = {"stage_8_22": (8, 22)}

MAKE_PLOTS = True

GENE_ANALYSIS = {
    "enabled": True,
    "cell_types": ["Neuroectoderm"],
    "score_col": "inter_timepoint_dissimilarity",
    "gene_q_threshold": 0.05,
    "label_top_n": 10,
    "label_species": "fish",
    "species": {
        "fish": {"h5ad": str(CODE_ROOT / "data" / "fish.h5ad"), "cell_id_col": None,
                 "is_preprocessed": True},
        "frog": {"h5ad": str(CODE_ROOT / "data" / "frog.h5ad"), "cell_id_col": None,
                 "is_preprocessed": True},
    },
    "homology_path": str(CODE_ROOT / "data" / "fish_frog_homology.csv"),
    "homology_columns": {"fish": "fish_gene", "frog": "frog_gene"},
    "go_species": "fish",
    "organism": "zebrafish",
    "gmt_path": CODE_ROOT / "data" / "go_bp.gmt",
    "obo_path": None,
    "download_go_names": True,
    "go_fdr": 0.05,
}


if __name__ == "__main__":
    run_developmental_dissimilarity_analysis(
        embedding_h5ad=EMBEDDING_H5AD,
        output_dir=OUTPUT_DIR,
        model_path=MODEL_PATH,
        training_config_path=TRAINING_CONFIG_PATH,
        device=DEVICE,
        batch_size=INFERENCE_BATCH_SIZE,
        species_metadata=SPECIES_METADATA,
        species_pair=SPECIES_PAIR,
        species_col=SPECIES_COL,
        time_col=TIME_COL,
        cell_id_col=CELL_ID_COL,
        cell_type_col=CELL_TYPE_COL,
        embedding_key=EMBEDDING_KEY,
        similarity_matrix_path=SIMILARITY_MATRIX_PATH,
        similarity_matrix_method=SIMILARITY_MATRIX_METHOD,
        alignment_ranges=ALIGNMENT_RANGES,
        alignment_transpose_axes=ALIGNMENT_TRANSPOSE_AXES,
        alignment_query_axis=ALIGNMENT_QUERY_AXIS,
        dtw_open_end=DTW_OPEN_END,
        time_orders=TIME_ORDERS,
        make_plots=MAKE_PLOTS,
        gene_analysis=GENE_ANALYSIS,
    )
