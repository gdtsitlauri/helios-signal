args = argv();
if numel(args) < 1
  disp("HELIOS Octave signal generation ready");
else
  outdir = args{1};
  if ~exist(outdir, "dir")
    system(sprintf("mkdir -p '%s'", outdir));
  end

  t = linspace(0, 1, 512)';
  chirp_signal = sin(2 * pi * (5 + 20 * t.^2) .* t);
  am_signal = (1 + 0.5 * sin(2 * pi * 3 * t)) .* sin(2 * pi * 40 * t);
  fm_signal = sin(2 * pi * 40 * t + 2 * sin(2 * pi * 3 * t));
  awgn = 0.1 * randn(size(t));
  colored = filter(1, [1 -0.95], awgn);

  signal_data = [t, chirp_signal, am_signal, fm_signal, awgn, colored];
  save("-ascii", [outdir "/generated_signals.csv"], "signal_data");
  disp("HELIOS Octave signal generation ready");
end
