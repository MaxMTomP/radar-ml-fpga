"""
Quantise floating-point MLP weights and biases to fixed-point format (Q1.7)
for FPGA implementation.

Loads weights from:
    data/ml_weights_bias/

Outputs quantised integer weights to:
    data/ml_weights_bias/quantised/

Also exports .txt versions for Verilog/VHDL.
"""

import os
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
SCALE = 128  # Q1.7 scaling factor
INT_MIN = -128
INT_MAX = 127


def quantise_array(arr: np.ndarray) -> np.ndarray:
    """Quantise a floating-point array to signed 8-bit integers using Q1.7."""
    scaled = arr * SCALE
    rounded = np.round(scaled)
    clipped = np.clip(rounded, INT_MIN, INT_MAX)
    return clipped.astype(np.int8)


def save_txt(array: np.ndarray, filepath: str):
    """Save array to a .txt file for HDL use (comma-separated)."""
    with open(filepath, "w") as f:
        if array.ndim == 1:
            f.write(", ".join(str(int(x)) for x in array))
        else:
            for row in array:
                f.write(", ".join(str(int(x)) for x in row) + "\n")


def main():
    # Input directory (float weights)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "../../data/ml_weights_bias")

    # Output directory (quantised)
    out_dir = os.path.join(base_dir, "quantised")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading weights from: {base_dir}")
    print(f"Saving quantised weights to: {out_dir}")

    # Load float weights
    W_h = np.load(os.path.join(base_dir, "ml_weights_hidden.npy"))
    b_h = np.load(os.path.join(base_dir, "ml_biases_hidden.npy"))
    W_o = np.load(os.path.join(base_dir, "ml_weights_output.npy"))
    b_o = np.load(os.path.join(base_dir, "ml_biases_output.npy"))

    # Quantise
    Q_W_h = quantise_array(W_h)
    Q_b_h = quantise_array(b_h)
    Q_W_o = quantise_array(W_o)
    Q_b_o = quantise_array(b_o)

    # Save as .npy
    np.save(os.path.join(out_dir, "Q_weights_hidden.npy"), Q_W_h)
    np.save(os.path.join(out_dir, "Q_biases_hidden.npy"), Q_b_h)
    np.save(os.path.join(out_dir, "Q_weights_output.npy"), Q_W_o)
    np.save(os.path.join(out_dir, "Q_biases_output.npy"), Q_b_o)

    # Save as .txt (HDL-friendly)
    save_txt(Q_W_h, os.path.join(out_dir, "Q_weights_hidden.txt"))
    save_txt(Q_b_h, os.path.join(out_dir, "Q_biases_hidden.txt"))
    save_txt(Q_W_o, os.path.join(out_dir, "Q_weights_output.txt"))
    save_txt(Q_b_o, os.path.join(out_dir, "Q_biases_output.txt"))

    print("Quantisation complete.")
    print("Files generated:")
    for file in os.listdir(out_dir):
        print("  -", file)


if __name__ == "__main__":
    main()
