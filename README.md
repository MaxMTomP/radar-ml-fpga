# radar-ml-fpga
Utilising Machine Learning tools to help weight the onboard neural networks on our custom FPGA system to deliver decision-critical information at ultra-low latencies. 

## Hardware building blocks
The `verilog/` directory contains synthesizable SystemVerilog modules that implement key neural-network accelerator blocks for radar signal processing:

| Module | Description |
| --- | --- |
| `mac_unit.sv` | Fixed-point multiply–accumulate with pipelined multiplier and configurable fractional precision. Includes a simple self-checking testbench. |
| `relu_activation.sv` | Signed ReLU activation with valid/ready style timing and accompanying testbench. |
| `weight_bias_regfile.sv` | Configurable weight and bias register file with independent write controls plus a smoke-test bench. |
| `stream_register.sv` | One-stage ready/valid stream register that can wrap existing datapaths and an illustrative testbench. |
| `neural_network_top.sv` | Example top-level pipeline that chains MAC and ReLU layers, backed by the register file for weights/biases, with a streaming interface and demo testbench. |


Each file includes an example testbench that can be simulated with your preferred SystemVerilog-capable simulator (e.g., `iverilog` or `xsim`).

These designs are intended for Intel Cyclone V (5CSEMA5F31C6) devices; synthesis and fitting can be performed with recent Quartus Prime releases.

