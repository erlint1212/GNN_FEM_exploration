# config.py
import torch
import dolfin # For dolfin.pi if needed here, though typically used in fem_utils

# --- Global Training Parameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 8
NUM_SAMPLES = 100 # Number of mesh samples for the dataset

# --- Mesh Type Selector ---
# Choose 'square' or 'pipe'
MESH_TYPE = 'square'

# --- FEM Problem Parameters (Poisson Equation) ---
F_EXPRESSION_STR_SQUARE = "2*pow(user_pi,2)*sin(user_pi*x[0])*sin(user_pi*x[1])"
U_EXACT_EXPRESSION_STR_SQUARE = "sin(user_pi*x[0])*sin(user_pi*x[1])"
U_DIRICHLET_EXPRESSION_STR_SQUARE = "0.0"
EXACT_SOL_DEGREE_SQUARE = 5

F_EXPRESSION_STR_PIPE = "10 * exp(-(pow(x[0] - 0.5*L, 2) + pow(x[1] - 0.5*H, 2)) / (2*pow(0.1*H,2)))"
U_EXACT_EXPRESSION_STR_PIPE = None
U_DIRICHLET_EXPRESSION_STR_PIPE = "0.0"
EXACT_SOL_DEGREE_PIPE = 5

# --- Default values for parameters that might be conditionally defined ---
MESH_SIZE_FACTOR_MIN = None # Default for pipe, defined if MESH_TYPE=='pipe'
MESH_SIZE_FACTOR_MAX = None # Default for pipe, defined if MESH_TYPE=='pipe'
PIPE_LENGTH = 3.0           # Default, used if MESH_TYPE=='pipe'
PIPE_HEIGHT = 1.0           # Default, used if MESH_TYPE=='pipe'
OBSTACLE_CENTER_X_FACTOR = 0.3 # Default, used if MESH_TYPE=='pipe'
OBSTACLE_CENTER_Y_FACTOR = 0.5 # Default, used if MESH_TYPE=='pipe'
OBSTACLE_RADIUS_FACTOR = 0.15 # Default, used if MESH_TYPE=='pipe'

# --- Parameters specific to MESH_TYPE ---
if MESH_TYPE == 'square':
    MODEL_NAME_SUFFIX_BASE = "SquareMesh_Poisson"
    MESH_SIZE_MIN = 8
    MESH_SIZE_MAX = 20
    # MESH_SIZE_FACTOR_MIN/MAX are not used for square, defaults above are fine
elif MESH_TYPE == 'pipe':
    MODEL_NAME_SUFFIX_BASE = "PipeObstacleMesh_Poisson"
    MESH_SIZE_FACTOR_MIN = 0.08 # Specific value for pipe
    MESH_SIZE_FACTOR_MAX = 0.20 # Specific value for pipe
    # PIPE_LENGTH, etc., use defaults or can be overridden here if needed
else:
    raise ValueError(f"Unknown MESH_TYPE: {MESH_TYPE}. Choose 'square' or 'pipe'.")

# --- GNN Model Hyperparameters ---
HIDDEN_CHANNELS = 128
OUT_CHANNELS = 2
HEADS = 8
NUM_LAYERS = 4
DROPOUT = 0.5

# --- Feature Engineering Switch ---
USE_MONITOR_AS_FEATURE = True

# --- Derived Names ---
MODEL_NAME_SUFFIX = f"{MODEL_NAME_SUFFIX_BASE}{'_FeatEngV2' if USE_MONITOR_AS_FEATURE else ''}" # Using V2 from your main script name
MODEL_NAME = f"RAdaptGAT_{MODEL_NAME_SUFFIX}"
BASE_OUTPUT_DIR = f"gat_{MODEL_NAME_SUFFIX.lower()}_outputs"

# --- R-Adaptivity Parameters ---
CLASSICAL_ADAPT_STRENGTH = 0.05

# --- Validation/Testing Parameters ---
NUM_INFERENCE_RUNS_PER_SAMPLE_FOR_TIMING = 5
