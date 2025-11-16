// Auto-generated SystemVerilog weight package
// Do not edit manually

package nn_weights;

    // Hidden layer weights: 6 neurons x 8 inputs
    localparam int W_HIDDEN [0:5][0:7] = '{
        '{ 94, 127, -64, 117, -128, 3 },
        '{ 41, 127, -47, 127, -128, 127 },
        '{ 127, 109, -128, 71, -128, 69 },
        '{ -128, 127, -91, -125, -87, 37 },
        '{ -126, 127, -128, -82, -43, -15 },
        '{ -66, 87, -128, 127, 127, -51 },
        '{ -120, -128, -96, 93, 14, -88 },
        '{ -125, -89, -128, 127, 50, -92 }
    };

    // Hidden layer biases
    localparam int B_HIDDEN [0:5] = '{ 126, 12, 127, 49, 127, 106 };

    // Output layer weights: 4 neurons x 6 inputs
    localparam int W_OUTPUT [0:3][0:5] = '{
        '{ 127, 127, -128, -128 },
        '{ -128, 1, 121, -128 },
        '{ 127, -60, -128, -102 },
        '{ -128, 127, -128, 127 },
        '{ 127, -128, -20, 127 },
        '{ 89, 80, 80, -128 }
    };

    // Output layer biases
    localparam int B_OUTPUT [0:3] = '{ -7, -128, 127, -10 };

endpackage : nn_weights
