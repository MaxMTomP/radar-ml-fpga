`timescale 1ns/1ps

module relu_activation #(
    parameter integer DATA_WIDTH = 16
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         ce,
    input  logic                         valid_in,
    input  logic signed [DATA_WIDTH-1:0] in_data,
    output logic                         valid_out,
    output logic signed [DATA_WIDTH-1:0] out_data
);
    always_ff @(posedge clk) begin
        if (rst) begin
            out_data  <= '0;
            valid_out <= 1'b0;
        end else if (ce) begin
            if (valid_in) begin
                if (in_data[DATA_WIDTH-1]) begin
                    out_data <= '0;
                end else begin
                    out_data <= in_data;
                end
            end
            valid_out <= valid_in;
        end
    end
endmodule

module tb_relu_activation;
    localparam integer DATA_WIDTH = 16;

    logic clk = 1'b0;
    logic rst = 1'b1;
    logic ce  = 1'b1;

    logic        valid_in = 1'b0;
    logic signed [DATA_WIDTH-1:0] in_data = '0;
    logic signed [DATA_WIDTH-1:0] out_data;
    logic        valid_out;

    relu_activation #(
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .ce(ce),
        .valid_in(valid_in),
        .in_data(in_data),
        .valid_out(valid_out),
        .out_data(out_data)
    );

    always #5 clk = ~clk;

    initial begin
        $display("ReLU activation testbench start");
        repeat (3) @(posedge clk);
        rst <= 1'b0;

        @(posedge clk);
        valid_in <= 1'b1;
        in_data  <= -16'sd128;

        @(posedge clk);
        valid_in <= 1'b1;
        in_data  <= 16'sd1024;

        @(posedge clk);
        valid_in <= 1'b1;
        in_data  <= 16'sd0;

        @(posedge clk);
        valid_in <= 1'b0;
        in_data  <= 16'sd0;

        repeat (4) @(posedge clk);
        $display("ReLU activation testbench end");
        $finish;
    end

    always_ff @(posedge clk) begin
        if (valid_out) begin
            $display("[%0t] out_data = %0d", $time, out_data);
        end
    end
endmodule
