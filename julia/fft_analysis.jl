using CSV
using DataFrames
using FFTW

function run_demo()
    signal = sin.(2pi .* 5 .* range(0, 1, length=128))
    spec = rfft(signal)
    println("HELIOS Julia FFT ready: ", length(spec), " bins")
end

function run_benchmark(input_path::String, output_path::String)
    df = CSV.read(input_path, DataFrame)
    signal = Vector{Float64}(df.signal)
    t0 = time()
    for _ in 1:50
        rfft(signal)
    end
    elapsed = (time() - t0) / 50
    out = DataFrame(backend=["julia"], seed=[0], signal_length=[length(signal)], seconds=[elapsed])
    CSV.write(output_path, out)
end

if length(ARGS) < 2
    run_demo()
else
    run_benchmark(ARGS[1], ARGS[2])
end
