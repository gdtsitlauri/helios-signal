using CSV
using DataFrames
using Statistics
using Wavelets

function snr(clean, test)
    err = clean .- test
    return 10 * log10(mean(clean .^ 2) / (mean(err .^ 2) + 1e-12))
end

function run_demo()
    println("HELIOS Julia wavelet pipeline ready")
end

function run_wavelet(input_path::String, output_path::String)
    df = CSV.read(input_path, DataFrame)
    clean = Vector{Float64}(df.clean)
    noisy = Vector{Float64}(df.noisy)
    wt = wavelet(WT.haar)
    coeffs = dwt(noisy, wt)
    thresh = median(abs.(coeffs)) * sqrt(2 * log(length(coeffs) + 1))
    shrunk = sign.(coeffs) .* max.(abs.(coeffs) .- thresh, 0)
    denoised = idwt(shrunk, wt)
    out = DataFrame(signal=["synthetic"], snr_before_db=[snr(clean, noisy)], snr_after_db=[snr(clean, denoised[1:length(clean)])])
    CSV.write(output_path, out)
end

if length(ARGS) < 2
    run_demo()
else
    run_wavelet(ARGS[1], ARGS[2])
end
