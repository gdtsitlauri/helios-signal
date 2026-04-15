using DSP

b = digitalfilter(Lowpass(0.2), Butterworth(4))
println("HELIOS Julia filter ready")
