`timescale 1ns/1ps

module stream_register #(
    parameter integer DATA_WIDTH = 32
) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  ce,
    input  wire                  in_valid,
    output wire                  in_ready,
    input  wire [DATA_WIDTH-1:0] in_data,
    output reg                   out_valid,
    input  wire                  out_ready,
    output reg  [DATA_WIDTH-1:0] out_data
);
    assign in_ready = ce ? (~out_valid || out_ready) : 1'b0;

    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            out_data  <= {DATA_WIDTH{1'b0}};
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

    reg clk = 1'b0;
    reg rst = 1'b1;
    reg ce  = 1'b1;

    reg  in_valid = 1'b0;
    reg  [DATA_WIDTH-1:0] in_data = 0;
    wire in_ready;

    wire out_valid;
    reg  out_ready = 1'b0;
    wire [DATA_WIDTH-1:0] out_data;

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
        in_valid <= 1'b1;
        in_data  <= 16'h0001;
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

    always @(posedge clk) begin
        $display("[%0t] in_valid=%b in_ready=%b out_valid=%b out_ready=%b out_data=%0h",
                 $time, in_valid, in_ready, out_valid, out_ready, out_data);
    end
endmodule
