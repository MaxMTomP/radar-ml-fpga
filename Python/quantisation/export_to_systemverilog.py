"""
Export quantised MLP weights into a SystemVerilog package file (nn_weights.svh).

This script reads the fixed-point integer weights from:
data/ml_weights_bias/quantised/

And generates:
verilog/nn_weights.svh

SystemVerilog Format:
package nn_weights;
    localparam int W_HIDDEN [0:5][0:7] = '{ ... };
    localparam int B_HIDDEN [0:5]      = '{ ... };
    localparam int W_OUTPUT [0:3][0:5] = '{ ... };
    localparam int B_OUTPUT [0:3]      = '{ ... };
endpackage : nn_weights
"""

import os
import numpy as np


def array_to_sv_literal(arr, indent="    "):
    """
    Convert a numpy array into SystemVerilog literal format.
    Handles both 1D and 2D arrays.
    """
    if arr.ndim == 1:
        # Example: '{ 1, -2, 3, 4 }
        elements = ", ".join(str(int(v)) for v in arr)
        return f"'{{ {elements} }}"

    elif arr.ndim == 2:
        # Example for 2D:
        # '{
        #     '{ 1, 2, 3 },
        #     '{ 4, 5, 6 }
        # }
        rows = []
        for row in arr:
            row_str = ", ".join(str(int(v)) for v in row)
            rows.append(f"{indent}    '{{ {row_str} }}")
        return "'{\n" + ",\n".join(rows) + f"\n{indent}}}"

    else:
        raise ValueError("Array must be 1D or 2D for SystemVerilog export.")


def main():

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    q_dir = os.path.join(script_dir, "../../data/ml_weights_bias/quantised")
    out_dir = os.path.join(script_dir, "../../verilog")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading quantised weights from:", q_dir)

    # Load quantised arrays
    W_hidden = np.load(os.path.join(q_dir, "Q_weights_hidden.npy"))
    b_hidden = np.load(os.path.join(q_dir, "Q_biases_hidden.npy"))
    W_output = np.load(os.path.join(q_dir, "Q_weights_output.npy"))
    b_output = np.load(os.path.join(q_dir, "Q_biases_output.npy"))

    # Convert to SystemVerilog literals
    W_hidden_sv = array_to_sv_literal(W_hidden)
    B_hidden_sv = array_to_sv_literal(b_hidden)
    W_output_sv = array_to_sv_literal(W_output)
    B_output_sv = array_to_sv_literal(b_output)

    # Output file
    out_path = os.path.join(out_dir, "nn_weights.svh")

    print("Writing SystemVerilog package to:", out_path)

    with open(out_path, "w") as f:
        f.write("// Auto-generated SystemVerilog weight package\n")
        f.write("// Do not edit manually\n\n")
        f.write("package nn_weights;\n\n")

        # Hidden weights
        f.write("    // Hidden layer weights: 6 neurons x 8 inputs\n")
        f.write("    localparam int W_HIDDEN [0:5][0:7] = ")
        f.write(W_hidden_sv + ";\n\n")

        # Hidden biases
        f.write("    // Hidden layer biases\n")
        f.write("    localparam int B_HIDDEN [0:5] = ")
        f.write(B_hidden_sv + ";\n\n")

        # Output weights
        f.write("    // Output layer weights: 4 neurons x 6 inputs\n")
        f.write("    localparam int W_OUTPUT [0:3][0:5] = ")
        f.write(W_output_sv + ";\n\n")

        # Output biases
        f.write("    // Output layer biases\n")
        f.write("    localparam int B_OUTPUT [0:3] = ")
        f.write(B_output_sv + ";\n\n")

        f.write("endpackage : nn_weights\n")

    print("Done. SystemVerilog weights exported successfully.")


if __name__ == "__main__":
    main()
