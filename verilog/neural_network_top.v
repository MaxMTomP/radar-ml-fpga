`timescale 1ns/1ps

module neural_network_top #(
    parameter integer DATA_WIDTH        = 16,
    parameter integer ACC_WIDTH         = 32,
    parameter integer FRAC_BITS         = 8,
    parameter integer LAYER_COUNT       = 3,
    parameter integer LAYER_ADDR_WIDTH  = 2
) (
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          ce,
    input  wire                          in_valid,
    output wire                          in_ready,
    input  wire signed [DATA_WIDTH-1:0]  in_data,
    output wire                          out_valid,
    output wire signed [DATA_WIDTH-1:0]  out_data,
    input  wire                          cfg_weight_we,
    input  wire [LAYER_ADDR_WIDTH-1:0]   cfg_weight_addr,
    input  wire [DATA_WIDTH-1:0]         cfg_weight_data,
    input  wire                          cfg_bias_we,
    input  wire [LAYER_ADDR_WIDTH-1:0]   cfg_bias_addr,
    input  wire [DATA_WIDTH-1:0]         cfg_bias_data,
    input  wire                          cfg_commit,
    output wire                          cfg_busy
);
    initial begin
        if (LAYER_COUNT <= 0) begin
            $error("LAYER_COUNT must be greater than zero");
        end
    end

    wire input_stage_valid;
    wire signed [DATA_WIDTH-1:0] input_stage_data;

    stream_register #(
        .DATA_WIDTH(DATA_WIDTH)
    ) input_buffer (
        .clk(clk),
        .rst(rst),
        .ce(ce),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_data(in_data),
        .out_valid(input_stage_valid),
        .out_ready(1'b1),
        .out_data(input_stage_data)
    );

    reg signed [DATA_WIDTH-1:0] layer_weight_reg [0:LAYER_COUNT-1];
    reg signed [DATA_WIDTH-1:0] layer_bias_reg   [0:LAYER_COUNT-1];

    reg [LAYER_ADDR_WIDTH-1:0] copy_addr;
    reg                        copy_active;
    reg                        copy_phase;

    integer i;
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < LAYER_COUNT; i = i + 1) begin
                layer_weight_reg[i] <= {DATA_WIDTH{1'b0}};
                layer_bias_reg[i]   <= {DATA_WIDTH{1'b0}};
            end
            copy_addr   <= {LAYER_ADDR_WIDTH{1'b0}};
            copy_active <= 1'b0;
            copy_phase  <= 1'b0;
        end else if (ce) begin
            if (!copy_active) begin
                if (cfg_commit) begin
                    copy_active <= 1'b1;
                    copy_phase  <= 1'b0;
                    copy_addr   <= {LAYER_ADDR_WIDTH{1'b0}};
                end
            end else begin
                if (!copy_phase) begin
                    copy_phase <= 1'b1;
                end else begin
                    layer_weight_reg[copy_addr] <= weight_read_data;
                    layer_bias_reg[copy_addr]   <= bias_read_data;
                    if (copy_addr == LAYER_COUNT-1) begin
                        copy_active <= 1'b0;
                        copy_phase  <= 1'b0;
                    end else begin
                        copy_addr  <= copy_addr + 1'b1;
                        copy_phase <= 1'b0;
                    end
                end
            end
        end
    end

    assign cfg_busy = copy_active;

    wire [DATA_WIDTH-1:0] weight_read_data;
    wire [DATA_WIDTH-1:0] bias_read_data;

    weight_bias_regfile #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_COUNT(LAYER_COUNT),
        .BIAS_COUNT(LAYER_COUNT),
        .WEIGHT_ADDR_WIDTH(LAYER_ADDR_WIDTH),
        .BIAS_ADDR_WIDTH(LAYER_ADDR_WIDTH)
    ) config_store (
        .clk(clk),
        .rst(rst),
        .ce(ce),
        .weight_we(cfg_weight_we),
        .weight_write_addr(cfg_weight_addr),
        .weight_write_data(cfg_weight_data),
        .bias_we(cfg_bias_we),
        .bias_write_addr(cfg_bias_addr),
        .bias_write_data(cfg_bias_data),
        .weight_read_addr(copy_addr),
        .bias_read_addr(copy_addr),
        .weight_read_data(weight_read_data),
        .bias_read_data(bias_read_data)
    );

    function signed [DATA_WIDTH-1:0] acc_to_data;
        input signed [ACC_WIDTH-1:0] value;
        begin
            acc_to_data = value >>> FRAC_BITS;
        end
    endfunction

    wire signed [DATA_WIDTH-1:0] layer_data [0:LAYER_COUNT];
    wire                         layer_valid [0:LAYER_COUNT];

    assign layer_data[0]  = input_stage_data;
    assign layer_valid[0] = input_stage_valid;

    genvar layer_idx;
    generate
        for (layer_idx = 0; layer_idx < LAYER_COUNT; layer_idx = layer_idx + 1) begin : g_layers
            wire signed [ACC_WIDTH-1:0] mac_acc;
            wire                        mac_valid;
            wire signed [ACC_WIDTH-1:0] bias_extended = $signed(layer_bias_reg[layer_idx]) <<< FRAC_BITS;
            wire signed [DATA_WIDTH-1:0] mac_scaled = acc_to_data(mac_acc);

            mac_unit #(
                .DATA_WIDTH(DATA_WIDTH),
                .ACC_WIDTH(ACC_WIDTH),
                .FRAC_BITS(FRAC_BITS)
            ) mac_inst (
                .clk(clk),
                .rst(rst),
                .ce(ce),
                .valid_in(layer_valid[layer_idx]),
                .a(layer_data[layer_idx]),
                .b(layer_weight_reg[layer_idx]),
                .acc_in(bias_extended),
                .valid_out(mac_valid),
                .acc_out(mac_acc)
            );

            relu_activation #(
                .DATA_WIDTH(DATA_WIDTH)
            ) relu_inst (
                .clk(clk),
                .rst(rst),
                .ce(ce),
                .valid_in(mac_valid),
                .in_data(mac_scaled),
                .valid_out(layer_valid[layer_idx+1]),
                .out_data(layer_data[layer_idx+1])
            );
        end
    endgenerate

    assign out_valid = layer_valid[LAYER_COUNT];
    assign out_data  = layer_data[LAYER_COUNT];
endmodule

module tb_neural_network_top;
    localparam integer DATA_WIDTH       = 16;
    localparam integer ACC_WIDTH        = 32;
    localparam integer FRAC_BITS        = 8;
    localparam integer LAYER_COUNT      = 3;
    localparam integer LAYER_ADDR_WIDTH = 2;

    reg clk = 1'b0;
    reg rst = 1'b1;
    reg ce  = 1'b1;

    reg  in_valid = 1'b0;
    wire in_ready;
    reg  signed [DATA_WIDTH-1:0] in_data = 0;
    wire out_valid;
    wire signed [DATA_WIDTH-1:0] out_data;

    reg cfg_weight_we = 1'b0;
    reg cfg_bias_we   = 1'b0;
    reg [LAYER_ADDR_WIDTH-1:0] cfg_weight_addr = 0;
    reg [LAYER_ADDR_WIDTH-1:0] cfg_bias_addr   = 0;
    reg [DATA_WIDTH-1:0] cfg_weight_data = 0;
    reg [DATA_WIDTH-1:0] cfg_bias_data   = 0;
    reg cfg_commit = 1'b0;
    wire cfg_busy;

    neural_network_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .FRAC_BITS(FRAC_BITS),
        .LAYER_COUNT(LAYER_COUNT),
        .LAYER_ADDR_WIDTH(LAYER_ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .ce(ce),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_data(in_data),
        .out_valid(out_valid),
        .out_data(out_data),
        .cfg_weight_we(cfg_weight_we),
        .cfg_weight_addr(cfg_weight_addr),
        .cfg_weight_data(cfg_weight_data),
        .cfg_bias_we(cfg_bias_we),
        .cfg_bias_addr(cfg_bias_addr),
        .cfg_bias_data(cfg_bias_data),
        .cfg_commit(cfg_commit),
        .cfg_busy(cfg_busy)
    );

    always #5 clk = ~clk;

    task write_weight;
        input [LAYER_ADDR_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] data;
        begin
            cfg_weight_addr <= addr;
            cfg_weight_data <= data;
            cfg_weight_we   <= 1'b1;
            @(posedge clk);
            cfg_weight_we   <= 1'b0;
        end
    endtask

    task write_bias;
        input [LAYER_ADDR_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] data;
        begin
            cfg_bias_addr <= addr;
            cfg_bias_data <= data;
            cfg_bias_we   <= 1'b1;
            @(posedge clk);
            cfg_bias_we   <= 1'b0;
        end
    endtask

    initial begin
        $display("Neural network top testbench start");
        repeat (4) @(posedge clk);
        rst <= 1'b0;

        write_weight(2'd0, 16'sd256);  // 1.0
        write_weight(2'd1, 16'sd128);  // 0.5
        write_weight(2'd2, 16'sd512);  // 2.0

        write_bias(2'd0, 16'sd0);
        write_bias(2'd1, 16'sd0);
        write_bias(2'd2, 16'sd0);

        @(posedge clk);
        cfg_commit <= 1'b1;
        @(posedge clk);
        cfg_commit <= 1'b0;

        wait(!cfg_busy);

        @(posedge clk);
        if (in_ready) begin
            in_valid <= 1'b1;
            in_data  <= 16'sd256;
        end

        @(posedge clk);
        in_valid <= 1'b0;
        in_data  <= 16'sd0;

        repeat (12) @(posedge clk);

        @(posedge clk);
        if (in_ready) begin
            in_valid <= 1'b1;
            in_data  <= -16'sd256;
        end

        @(posedge clk);
        in_valid <= 1'b0;
        in_data  <= 16'sd0;

        repeat (12) @(posedge clk);
        $display("Neural network top testbench end");
        $finish;
    end

    always @(posedge clk) begin
        if (out_valid) begin
            $display("[%0t] Output sample = %0d", $time, out_data);
        end
    end
endmodule
