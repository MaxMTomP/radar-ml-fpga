`timescale 1ns/1ps

module stream_register #(
    parameter integer DATA_WIDTH = 32
) (
    input  logic                  clk,
    input  logic                  rst,
    input  logic                  ce,
    input  logic                  in_valid,
    output logic                  in_ready,
    input  logic [DATA_WIDTH-1:0] in_data,
    output logic                  out_valid,
    input  logic                  out_ready,
    output logic [DATA_WIDTH-1:0] out_data
);
    assign in_ready = ce ? (~out_valid || out_ready) : 1'b0;

    always_ff @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            out_data  <= '0;
        end else if (ce) begin
            case ({in_valid && in_ready, out_valid && out_ready})
                2'b00: begin
                    out_valid <= out_valid;
                end
                2'b01: begin
                    out_valid <= 1'b0;
                end
                2'b10: begin
                    out_valid <= 1'b1;
                    out_data  <= in_data;
                end
                2'b11: begin
                    out_valid <= 1'b1;
                    out_data  <= in_data;
                end
            endcase
        end
    end
endmodule

module tb_stream_register;
    localparam integer DATA_WIDTH = 16;

    logic clk = 1'b0;
    logic rst = 1'b1;
    logic ce  = 1'b1;

    logic                  in_valid = 1'b0;
    logic [DATA_WIDTH-1:0] in_data  = '0;
    logic                  in_ready;

    logic                  out_valid;
    logic                  out_ready = 1'b0;
    logic [DATA_WIDTH-1:0] out_data;

    stream_register #(
        .DATA_WIDTH(DATA_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .ce(ce),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_data(in_data),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_data(out_data)
    );

    always #5 clk = ~clk;

    initial begin
        $display("Stream wrapper testbench start");
        repeat (2) @(posedge clk);
        rst <= 1'b0;

        @(posedge clk);
        in_valid  <= 1'b1;
        in_data   <= 16'h0001;
        out_ready <= 1'b0;

        @(posedge clk);
        out_ready <= 1'b1;

        @(posedge clk);
        in_data   <= 16'h0002;
        in_valid  <= 1'b1;
        out_ready <= 1'b1;

        @(posedge clk);
        in_valid  <= 1'b0;
        out_ready <= 1'b0;

        @(posedge clk);
        out_ready <= 1'b1;

        repeat (4) @(posedge clk);
        $display("Stream wrapper testbench end");
        $finish;
    end

    always_ff @(posedge clk) begin
        $display("[%0t] in_valid=%b in_ready=%b out_valid=%b out_ready=%b out_data=%0h",
                 $time, in_valid, in_ready, out_valid, out_ready, out_data);
    end
endmodule
