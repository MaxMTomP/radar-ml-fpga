`timescale 1ns/1ps

module weight_bias_regfile #(
    parameter integer DATA_WIDTH         = 16,
    parameter integer WEIGHT_COUNT       = 64,
    parameter integer BIAS_COUNT         = 16,
    parameter integer WEIGHT_ADDR_WIDTH  = 6,
    parameter integer BIAS_ADDR_WIDTH    = 4
) (
    input  logic                          clk,
    input  logic                          rst,
    input  logic                          ce,
    input  logic                          weight_we,
    input  logic [WEIGHT_ADDR_WIDTH-1:0]  weight_write_addr,
    input  logic signed [DATA_WIDTH-1:0]  weight_write_data,
    input  logic                          bias_we,
    input  logic [BIAS_ADDR_WIDTH-1:0]    bias_write_addr,
    input  logic signed [DATA_WIDTH-1:0]  bias_write_data,
    input  logic [WEIGHT_ADDR_WIDTH-1:0]  weight_read_addr,
    input  logic [BIAS_ADDR_WIDTH-1:0]    bias_read_addr,
    output logic signed [DATA_WIDTH-1:0]  weight_read_data,
    output logic signed [DATA_WIDTH-1:0]  bias_read_data
);
    logic signed [DATA_WIDTH-1:0] weight_mem [0:WEIGHT_COUNT-1];
    logic signed [DATA_WIDTH-1:0] bias_mem   [0:BIAS_COUNT-1];

    integer i;

    always_ff @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < WEIGHT_COUNT; i = i + 1) begin
                weight_mem[i] <= '0;
            end
            for (i = 0; i < BIAS_COUNT; i = i + 1) begin
                bias_mem[i] <= '0;
            end
            weight_read_data <= '0;
            bias_read_data   <= '0;
        end else if (ce) begin
            if (weight_we) begin
                weight_mem[weight_write_addr] <= weight_write_data;
            end
            if (bias_we) begin
                bias_mem[bias_write_addr] <= bias_write_data;
            end
            weight_read_data <= weight_mem[weight_read_addr];
            bias_read_data   <= bias_mem[bias_read_addr];
        end
    end
endmodule

module tb_weight_bias_regfile;
    localparam integer DATA_WIDTH        = 16;
    localparam integer WEIGHT_COUNT      = 4;
    localparam integer BIAS_COUNT        = 2;
    localparam integer WEIGHT_ADDR_WIDTH = 2;
    localparam integer BIAS_ADDR_WIDTH   = 1;

    logic clk = 1'b0;
    logic rst = 1'b1;
    logic ce  = 1'b1;

    logic                         weight_we = 1'b0;
    logic                         bias_we   = 1'b0;
    logic [WEIGHT_ADDR_WIDTH-1:0] weight_write_addr = '0;
    logic [BIAS_ADDR_WIDTH-1:0]   bias_write_addr   = '0;
    logic signed [DATA_WIDTH-1:0] weight_write_data = '0;
    logic signed [DATA_WIDTH-1:0] bias_write_data   = '0;
    logic [WEIGHT_ADDR_WIDTH-1:0] weight_read_addr  = '0;
    logic [BIAS_ADDR_WIDTH-1:0]   bias_read_addr    = '0;
    logic signed [DATA_WIDTH-1:0] weight_read_data;
    logic signed [DATA_WIDTH-1:0] bias_read_data;

    weight_bias_regfile #(
        .DATA_WIDTH(DATA_WIDTH),
        .WEIGHT_COUNT(WEIGHT_COUNT),
        .BIAS_COUNT(BIAS_COUNT),
        .WEIGHT_ADDR_WIDTH(WEIGHT_ADDR_WIDTH),
        .BIAS_ADDR_WIDTH(BIAS_ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .ce(ce),
        .weight_we(weight_we),
        .weight_write_addr(weight_write_addr),
        .weight_write_data(weight_write_data),
        .bias_we(bias_we),
        .bias_write_addr(bias_write_addr),
        .bias_write_data(bias_write_data),
        .weight_read_addr(weight_read_addr),
        .bias_read_addr(bias_read_addr),
        .weight_read_data(weight_read_data),
        .bias_read_data(bias_read_data)
    );

    always #5 clk = ~clk;

    initial begin
        $display("Register file testbench start");
        repeat (3) @(posedge clk);
        rst <= 1'b0;

        @(posedge clk);
        weight_we         <= 1'b1;
        weight_write_addr <= 2'd1;
        weight_write_data <= 16'h0100;
        bias_we           <= 1'b1;
        bias_write_addr   <= 1'd0;
        bias_write_data   <= 16'h000A;

        @(posedge clk);
        weight_we         <= 1'b1;
        weight_write_addr <= 2'd2;
        weight_write_data <= 16'h0200;
        bias_we           <= 1'b0;

        @(posedge clk);
        weight_we         <= 1'b0;
        weight_read_addr  <= 2'd1;
        bias_read_addr    <= 1'd0;

        @(posedge clk);
        weight_read_addr  <= 2'd2;

        repeat (4) @(posedge clk);
        $display("Register file testbench end");
        $finish;
    end

    always_ff @(posedge clk) begin
        if (!rst) begin
            $display("[%0t] weight_rd=%0h bias_rd=%0h", $time, weight_read_data, bias_read_data);
        end
    end
endmodule
