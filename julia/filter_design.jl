using CSV
using DSP
using DataFrames

function run_demo()
    response = digitalfilter(Lowpass(0.2), Butterworth(4))
    println("HELIOS Julia filter ready")
end

function run_filter(output_path::String)
    response = digitalfilter(Lowpass(0.2), Butterworth(4))
    h = freqresp(response, range(0, π; length=256))
    out = DataFrame(
        omega=collect(range(0, π; length=256)),
        magnitude=abs.(h),
        phase=angle.(h),
    )
    CSV.write(output_path, out)
end

if isempty(ARGS)
    run_demo()
else
    run_filter(ARGS[1])
end
