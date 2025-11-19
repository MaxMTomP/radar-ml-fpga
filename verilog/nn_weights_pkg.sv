`timescale 1ns/1ps

package nn_weights_pkg;
    localparam int DATA_WIDTH  = 16;
    localparam int LAYER_COUNT = 3;

    // Simple fixed-point coefficients that map low switch values to
    // clearly different LED/HEX output codes after the NN pipeline.
    localparam logic signed [DATA_WIDTH-1:0] DEFAULT_WEIGHTS [0:LAYER_COUNT-1] = '{
        16'sh0100,  // layer 0 weight = +1.00
        16'sh0080,  // layer 1 weight = +0.50
        16'shFF00   // layer 2 weight = -1.00
    };

    localparam logic signed [DATA_WIDTH-1:0] DEFAULT_BIASES [0:LAYER_COUNT-1] = '{
        16'sh0000,  // no bias on layer 0
        16'sh0100,  // add +1.00 after first activation
        16'shFF80   // subtract 0.50 before the final activation
    };
endpackage : nn_weights_pkg
