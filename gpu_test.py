import sys
import numpy as np

# --- Helper function to print results nicely ---
def print_status(component, is_success, message=""):
    """Prints a formatted status line."""
    if is_success:
        print(f"✅ [SUCCESS] {component}: {message}")
    else:
        print(f"❌ [FAILURE] {component}: {message}", file=sys.stderr)

# --- Test 1: TensorFlow ---
print("--- 1. Testing TensorFlow ---")
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print_status("TensorFlow", True, f"Found {len(gpus)} GPU(s): {gpus}")
        # Try a simple operation
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        print_status("TensorFlow", True, "Successfully executed a matrix multiplication on the GPU.")
    else:
        print_status("TensorFlow", False, "Could not find any GPU devices.")
except ImportError as e:
    print_status("TensorFlow", False, f"Import Error: {e}. Is TensorFlow installed?")
except Exception as e:
    print_status("TensorFlow", False, f"An unexpected error occurred: {e}")

print("\n" + "="*50 + "\n")

# --- Test 2: XGBoost ---
print("--- 2. Testing XGBoost ---")
try:
    import xgboost as xgb
    print_status("XGBoost", True, f"XGBoost version {xgb.__version__} imported successfully.")
    # Create dummy data
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    # Attempt to initialize a classifier with GPU settings
    xgb_gpu = xgb.XGBClassifier(tree_method='gpu_hist', device='cuda')
    print_status("XGBoost", True, "Successfully initialized XGBClassifier with tree_method='gpu_hist'.")
    # You can even do a quick fit
    xgb_gpu.fit(X, y)
    print_status("XGBoost", True, "Successfully completed a small .fit() cycle on the GPU.")
except ImportError as e:
    print_status("XGBoost", False, f"Import Error: {e}. Is XGBoost installed?")
except Exception as e:
    print_status("XGBoost", False, f"An unexpected error occurred: {e}")
    print("           This often means there's an issue with your CUDA driver or toolkit visibility.")

print("\n" + "="*50 + "\n")

# --- Test 3: LightGBM ---
print("--- 3. Testing LightGBM ---")
try:
    import lightgbm as lgb
    print_status("LightGBM", True, f"LightGBM version {lgb.__version__} imported successfully.")
    # Create dummy data
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    # Attempt to initialize a classifier with GPU settings
    lgb_gpu = lgb.LGBMClassifier(device='gpu')
    print_status("LightGBM", True, "Successfully initialized LGBMClassifier with device='gpu'.")
    # You can even do a quick fit
    lgb_gpu.fit(X, y)
    print_status("LightGBM", True, "Successfully completed a small .fit() cycle on the GPU.")
except ImportError as e:
    print_status("LightGBM", False, f"Import Error: {e}. Is LightGBM installed?")
except Exception as e:
    print_status("LightGBM", False, f"An unexpected error occurred: {e}")
    print("           This can happen if LightGBM was not compiled with GPU support.")

print("\n" + "="*50 + "\n")
