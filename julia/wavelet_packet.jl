using CSV
using DataFrames
using Wavelets

function wpt_decompose(signal::Vector{Float64}, wavelet_name::String, levels::Int)
    wt = wavelet(getfield(WT, Symbol(wavelet_name)))

    n = length(signal)
    num_nodes = 2^levels

    if n % num_nodes != 0
        error("Signal length ($n) must be divisible by 2^levels ($num_nodes)")
    end

    all_coeffs = wpt(signal, wt, levels)

    node_size = div(n, num_nodes)
    coeffs_list = Vector{Vector{Float64}}()

    for i in 1:num_nodes
        start_idx = (i - 1) * node_size + 1
        end_idx = i * node_size
        push!(coeffs_list, all_coeffs[start_idx:end_idx])
    end

    return coeffs_list
end

function run_wavelet_packet(input_path::String, output_path::String, wavelet_name::String, levels::Int)
    df = CSV.read(input_path, DataFrame)
    signal = Vector{Float64}(df.signal)

    coeffs_list = wpt_decompose(signal, wavelet_name, levels)

    out = DataFrame()
    for (i, c) in enumerate(coeffs_list)
        out[!, Symbol("node_$(i-1)")] = c
    end

    CSV.write(output_path, out)
end

if length(ARGS) == 4
    try
        run_wavelet_packet(ARGS[1], ARGS[2], ARGS[3], parse(Int, ARGS[4]))
    catch e
        @error "Julia WPT Error" exception=e
        exit(1)
    end
else
    println("Usage: julia wavelet_packet.jl <input.csv> <output.csv> <wavelet> <levels>")
end