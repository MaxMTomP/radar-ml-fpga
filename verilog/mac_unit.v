`timescale 1ns/1ps

module mac_unit #(
    parameter integer DATA_WIDTH = 16,
    parameter integer ACC_WIDTH  = 32,
    parameter integer FRAC_BITS  = 8
) (
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          ce,
    input  wire                          valid_in,
    input  wire signed [DATA_WIDTH-1:0]  a,
    input  wire signed [DATA_WIDTH-1:0]  b,
    input  wire signed [ACC_WIDTH-1:0]   acc_in,
    output reg                           valid_out,
    output reg  signed [ACC_WIDTH-1:0]   acc_out
);
    localparam integer PROD_WIDTH = DATA_WIDTH * 2;

    reg signed [PROD_WIDTH-1:0] mult_pipe;
    reg signed [ACC_WIDTH-1:0]  acc_pipe;
    reg                         valid_pipe;

    function [ACC_WIDTH-1:0] sign_resize;
        input signed [PROD_WIDTH-1:0] value;
        begin
            if (ACC_WIDTH >= PROD_WIDTH) begin
                sign_resize = {{(ACC_WIDTH-PROD_WIDTH){value[PROD_WIDTH-1]}}, value};
            end else begin
                sign_resize = value[PROD_WIDTH-1 -: ACC_WIDTH];
            end
        end
    endfunction

    wire signed [PROD_WIDTH-1:0] product_shifted = mult_pipe >>> FRAC_BITS;
    wire signed [ACC_WIDTH-1:0]  mult_scaled     = $signed(sign_resize(product_shifted));

    always @(posedge clk) begin
        if (rst) begin
            mult_pipe  <= {PROD_WIDTH{1'b0}};
            acc_pipe   <= {ACC_WIDTH{1'b0}};
            valid_pipe <= 1'b0;
        end else if (ce) begin
            if (valid_in) begin
                mult_pipe <= a * b;
                acc_pipe  <= acc_in;
            end
            valid_pipe <= valid_in;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            acc_out   <= {ACC_WIDTH{1'b0}};
            valid_out <= 1'b0;
        end else if (ce) begin
            if (valid_pipe) begin
                acc_out <= acc_pipe + mult_scaled;
            end
            valid_out <= valid_pipe;
        end
    end
endmodule

module tb_mac_unit;
    localparam integer DATA_WIDTH = 16;
    localparam integer ACC_WIDTH  = 32;
    localparam integer FRAC_BITS  = 8;

    reg clk = 1'b0;
    reg rst = 1'b1;
    reg ce  = 1'b1;

    reg  valid_in = 1'b0;
    reg  signed [DATA_WIDTH-1:0] a = 0;
    reg  signed [DATA_WIDTH-1:0] b = 0;
    reg  signed [ACC_WIDTH-1:0]  acc_in = 0;
    wire signed [ACC_WIDTH-1:0]  acc_out;
    wire valid_out;

    mac_unit #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH (ACC_WIDTH),
        .FRAC_BITS (FRAC_BITS)
    ) dut (
        .clk(clk),
        .rst(rst),
        .ce(ce),
        .valid_in(valid_in),
        .a(a),
        .b(b),
        .acc_in(acc_in),
        .valid_out(valid_out),
        .acc_out(acc_out)
    );

    always #5 clk = ~clk;

    initial begin
        $display("MAC unit testbench start");
        repeat (4) @(posedge clk);
        rst <= 1'b0;

        @(posedge clk);
        valid_in <= 1'b1;
        a        <= 16'sd256;   // 1.0 in Q8
        b        <= 16'sd512;   // 2.0 in Q8
        acc_in   <= 32'sd0;

        @(posedge clk);
        valid_in <= 1'b1;
        a        <= -16'sd256;  // -1.0 in Q8
        b        <= 16'sd256;   // 1.0 in Q8
        acc_in   <= 32'sd256;   // 1.0 in Q8

        @(posedge clk);
        valid_in <= 1'b0;
        a        <= 0;
        b        <= 0;
        acc_in   <= 0;

        repeat (6) @(posedge clk);
        $display("MAC unit testbench end");
        $finish;
    end

    always @(posedge clk) begin
        if (valid_out) begin
            $display("[%0t] acc_out = %0d", $time, acc_out);
        end
    end
endmodule
