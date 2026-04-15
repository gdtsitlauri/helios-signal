using FFTW

signal = sin.(2pi .* 5 .* range(0, 1, length=128))
spec = rfft(signal)
println("HELIOS Julia FFT ready: ", length(spec), " bins")
