# radar-ml-fpga
Utilising Machine Learning tools to help weight the onboard neural networks on our custom FPGA system to deliver decision-critical information at ultra-low latencies. 

## Hardware building blocks
The `verilog/` directory contains synthesizable Verilog-2001 modules that implement key neural-network accelerator blocks for radar signal processing:

| Module | Description |
| --- | --- |
| `mac_unit.v` | Fixed-point multiply–accumulate with pipelined multiplier and configurable fractional precision. Includes a simple self-checking testbench. |
| `relu_activation.v` | Signed ReLU activation with valid/ready style timing and accompanying testbench. |
| `weight_bias_regfile.v` | Configurable weight and bias register file with independent write controls plus a smoke-test bench. |
| `stream_register.v` | One-stage ready/valid stream register that can wrap existing datapaths and an illustrative testbench. |
| `neural_network_top.v` | Example top-level pipeline that chains MAC and ReLU layers, backed by the register file for weights/biases, with a streaming interface and demo testbench. |

Each file includes an example testbench that can be simulated with your preferred Verilog simulator (e.g., `iverilog` or `xsim`).
