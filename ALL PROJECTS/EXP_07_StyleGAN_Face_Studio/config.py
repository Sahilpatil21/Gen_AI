"""Central configuration (values taken from Exp_no_7_gen_ai.ipynb)."""
import torch

SEED = 42

# Data
DATASET_NAME = "tpremoli/CelebA-attrs-20k"
IMG_SIZE = 64
NUM_IMAGES = 20000
BATCH_SIZE = 64
NUM_WORKERS = 2

# Model
Z_DIM = 256
W_DIM = 256
MAPPING_LAYERS = 6

# Optimisation
LR_G = 2e-3
LR_D = 2e-3
MAPPING_LR_MULT = 0.01   # mapping network trains 100x slower than synthesis network
R1_GAMMA = 10.0
R1_INTERVAL = 16         # lazy R1 regularisation
ADAM_BETAS = (0.0, 0.99)

# Training
NUM_EPOCHS = 15
SAMPLE_EVERY = 100       # steps between console logs

# Paths
SAMPLES_DIR = "samples"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
