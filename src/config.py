from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "ds004504"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_GRAPHS = PROJECT_ROOT / "data" / "graphs"
RESULTS_DIR = PROJECT_ROOT / "results"

PARTICIPANTS_TSV = DATA_RAW / "participants.tsv"

# --- Preprocessing params ---
L_FREQ = 0.5
H_FREQ = 45.0
NOTCH_FREQ = 50.0
REFERENCE = "average"

AMPLITUDE_THRESHOLD = 150e-6

#TODO maybe change later in the final phase to check for better results
EPOCH_DURATION = 4.0
EPOCH_OVERLAP = 0.0

# --- qEEG frequency bands ---
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}

# --- Connectivity params---
CONNECTIVITY_METHOD = "pearson"
TOP_K_EDGES = 0.3

# --- Group labels ---
GROUP_MAP = {"A": 1, "C": 0}   # AD = 1, Control = 0 (F = FTD, excluded)

# --- GCN hyperparameters ---
GCN_HIDDEN_DIM = 32
GCN_NUM_LAYERS = 2
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
BATCH_SIZE = 16
RANDOM_SEED = 42
