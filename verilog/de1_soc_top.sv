`timescale 1ns/1ps

`include "nn_weights_pkg.sv"

module de1_soc_top (
    input  logic        CLOCK_50,
    input  logic [3:0]  KEY,
    input  logic [9:0]  SW,
    output logic [9:0]  LEDR,
    output logic [6:0]  HEX0,
    output logic [6:0]  HEX1
);
    localparam int NN_DATA_WIDTH       = nn_weights_pkg::DATA_WIDTH;
    localparam int NN_ACC_WIDTH        = 32;
    localparam int NN_FRAC_BITS        = 8;
    localparam int NN_LAYER_COUNT      = nn_weights_pkg::LAYER_COUNT;
    localparam int NN_LAYER_ADDR_WIDTH = (NN_LAYER_COUNT > 1) ? $clog2(NN_LAYER_COUNT) : 1;
    localparam int CFG_INDEX_WIDTH     = (NN_LAYER_COUNT > 1) ? $clog2(NN_LAYER_COUNT + 1) : 1;

    // Synchronize the asynchronous push-buttons before they control resets/config reloads.
    logic rst_meta, rst_sync;
    always_ff @(posedge CLOCK_50) begin
        rst_meta <= ~KEY[0];
        rst_sync <= rst_meta;
    end

    logic reload_meta, reload_sync;
    always_ff @(posedge CLOCK_50) begin
        reload_meta <= ~KEY[1];
        reload_sync <= reload_meta;
    end

    logic reload_sync_d;
    always_ff @(posedge CLOCK_50) begin
        if (rst_sync) begin
            reload_sync_d <= 1'b0;
        end else begin
            reload_sync_d <= reload_sync;
        end
    end

    wire reload_pulse = reload_sync & ~reload_sync_d;

    // Convert the ten switches into a signed fixed-point sample.
    logic signed [NN_DATA_WIDTH-1:0] nn_input_value;
    always_comb begin
        nn_input_value = {{(NN_DATA_WIDTH-10){SW[9]}}, SW};
    end

    // Configuration interface wires.
    logic                           cfg_weight_we;
    logic [NN_LAYER_ADDR_WIDTH-1:0] cfg_weight_addr;
    logic signed [NN_DATA_WIDTH-1:0] cfg_weight_data;
    logic                           cfg_bias_we;
    logic [NN_LAYER_ADDR_WIDTH-1:0] cfg_bias_addr;
    logic signed [NN_DATA_WIDTH-1:0] cfg_bias_data;
    logic                           cfg_commit;
    logic                           cfg_busy;

    // NN datapath I/O wires.
    logic                           nn_in_ready;
    logic                           nn_out_valid;
    logic signed [NN_DATA_WIDTH-1:0] nn_output_value;

    logic cfg_ready;

    neural_network_top #(
        .DATA_WIDTH(NN_DATA_WIDTH),
        .ACC_WIDTH(NN_ACC_WIDTH),
        .FRAC_BITS(NN_FRAC_BITS),
        .LAYER_COUNT(NN_LAYER_COUNT),
        .LAYER_ADDR_WIDTH(NN_LAYER_ADDR_WIDTH)
    ) nn_core (
        .clk(CLOCK_50),
        .rst(rst_sync),
        .ce(1'b1),
        .in_valid(cfg_ready),
        .in_ready(nn_in_ready),
        .in_data(nn_input_value),
        .out_valid(nn_out_valid),
        .out_data(nn_output_value),
        .cfg_weight_we(cfg_weight_we),
        .cfg_weight_addr(cfg_weight_addr),
        .cfg_weight_data(cfg_weight_data),
        .cfg_bias_we(cfg_bias_we),
        .cfg_bias_addr(cfg_bias_addr),
        .cfg_bias_data(cfg_bias_data),
        .cfg_commit(cfg_commit),
        .cfg_busy(cfg_busy)
    );

    typedef enum logic [1:0] {
        CFG_LOAD,
        CFG_COMMIT,
        CFG_WAIT,
        CFG_READY
    } cfg_state_t;

    cfg_state_t cfg_state_q, cfg_state_d;
    logic [CFG_INDEX_WIDTH-1:0] cfg_index_q, cfg_index_d;

    always_ff @(posedge CLOCK_50) begin
        if (rst_sync) begin
            cfg_state_q <= CFG_LOAD;
            cfg_index_q <= '0;
        end else begin
            cfg_state_q <= cfg_state_d;
            cfg_index_q <= cfg_index_d;
        end
    end

    always_comb begin
        cfg_state_d      = cfg_state_q;
        cfg_index_d      = cfg_index_q;
        cfg_weight_we    = 1'b0;
        cfg_bias_we      = 1'b0;
        cfg_weight_addr  = '0;
        cfg_bias_addr    = '0;
        cfg_weight_data  = '0;
        cfg_bias_data    = '0;
        cfg_commit       = 1'b0;
        cfg_ready        = 1'b0;

        case (cfg_state_q)
            CFG_LOAD: begin
                if (cfg_index_q < NN_LAYER_COUNT) begin
                    cfg_weight_we   = 1'b1;
                    cfg_bias_we     = 1'b1;
                    cfg_weight_addr = cfg_index_q[NN_LAYER_ADDR_WIDTH-1:0];
                    cfg_bias_addr   = cfg_index_q[NN_LAYER_ADDR_WIDTH-1:0];
                    cfg_weight_data = nn_weights_pkg::DEFAULT_WEIGHTS[cfg_index_q];
                    cfg_bias_data   = nn_weights_pkg::DEFAULT_BIASES[cfg_index_q];
                    cfg_index_d     = cfg_index_q + 1'b1;
                end else begin
                    cfg_state_d = CFG_COMMIT;
                end
            end
            CFG_COMMIT: begin
                cfg_commit  = 1'b1;
                cfg_state_d = CFG_WAIT;
            end
            CFG_WAIT: begin
                if (!cfg_busy) begin
                    cfg_state_d = CFG_READY;
                end
            end
            CFG_READY: begin
                cfg_ready = 1'b1;
                if (reload_pulse) begin
                    cfg_state_d = CFG_LOAD;
                    cfg_index_d = '0;
                end
            end
            default: begin
                cfg_state_d = CFG_LOAD;
                cfg_index_d = '0;
            end
        endcase
    end

    logic signed [NN_DATA_WIDTH-1:0] display_value;
    always_ff @(posedge CLOCK_50) begin
        if (rst_sync) begin
            display_value <= '0;
        end else if (nn_out_valid) begin
            display_value <= nn_output_value;
        end
    end

    // Drive LEDs: LEDR[9] shows the sign bit, the rest show magnitude.
    assign LEDR[9]   = display_value[NN_DATA_WIDTH-1];
    assign LEDR[8:0] = display_value[8:0];

    function automatic logic [6:0] hex_decode (input logic [3:0] value);
        case (value)
            4'h0: hex_decode = 7'b1000000;
            4'h1: hex_decode = 7'b1111001;
            4'h2: hex_decode = 7'b0100100;
            4'h3: hex_decode = 7'b0110000;
            4'h4: hex_decode = 7'b0011001;
            4'h5: hex_decode = 7'b0010010;
            4'h6: hex_decode = 7'b0000010;
            4'h7: hex_decode = 7'b1111000;
            4'h8: hex_decode = 7'b0000000;
            4'h9: hex_decode = 7'b0010000;
            4'hA: hex_decode = 7'b0001000;
            4'hB: hex_decode = 7'b0000011;
            4'hC: hex_decode = 7'b1000110;
            4'hD: hex_decode = 7'b0100001;
            4'hE: hex_decode = 7'b0000110;
            4'hF: hex_decode = 7'b0001110;
            default: hex_decode = 7'b1111111;
        endcase
    endfunction

    assign HEX0 = hex_decode(display_value[3:0]);
    assign HEX1 = hex_decode(display_value[7:4]);
endmodule
