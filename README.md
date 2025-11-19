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
| `nn_weights_pkg.sv` | Package that captures a small, signed fixed-point weight/bias set used to seed the demo design. |
| `de1_soc_top.sv` | FPGA-ready wrapper that maps the neural-network pipeline onto DE1-SoC inputs (switches, push buttons) and visible outputs (LEDs, seven-segment displays). |

Each file includes an example testbench that can be simulated with your preferred SystemVerilog-capable simulator (e.g., `iverilog` or `xsim`).

These designs are intended for Intel Cyclone V (5CSEMA5F31C6) devices; synthesis and fitting can be performed with recent Quartus Prime releases.

## Board-level demonstration
To satisfy coursework that requires observable FPGA I/O, use `verilog/de1_soc_top.sv` as the project top level together with the supplied weight package. The wrapper performs the following tasks automatically:

* loads the default weights/biases from `nn_weights_pkg.sv` into `neural_network_top` via the configuration interface and commits them before any samples are processed
* treats `SW[9:0]` as a signed fixed-point activation (SW9 is the sign bit) and continually feeds that value into the NN once configuration has finished
* exposes the most recent NN output on LEDR (sign on LEDR9, magnitude on LEDR8–0) and on HEX0/HEX1 (low/high nibble)
* lets KEY0 act as a synchronous system reset and KEY1 request a fresh weight reload without reprogramming the FPGA

### Usage steps
1. Create or open a DE1-SoC project targeting device 5CSEMA5F31C6 in Quartus Prime.
2. Add every file in `verilog/` to the project, set `de1_soc_top` as the top-level entity, and import the board’s default pin assignments if desired.
3. Compile, program the FPGA, and wait a second for the HEX displays to update once the configuration FSM finishes copying the weights into the NN pipeline (LEDR will show 0x00 until then).
4. Toggle SW0–SW8 to change the activation magnitude and observe the LEDs/HEX change immediately; flip SW9 to assert a negative sign; press KEY0 to reset the datapath or tap KEY1 to reload the demo weights.
