args = argv();

if numel(args) < 1
  disp("HELIOS Octave control systems ready");
else
  outdir = args{1};
  if ~exist(outdir, "dir")
    system(sprintf("mkdir -p '%s'", outdir));
  end

  w = (10 .^ linspace(-2, 2, 128))';
  H = 1 ./ (1 + 1i .* w);
  bode_data = [w(:), 20 * log10(abs(H(:))), angle(H(:)) * 180 / pi];
  save("-ascii", [outdir "/bode.csv"], "bode_data");

  nyquist_data = [w(:), real(H(:)), imag(H(:))];
  save("-ascii", [outdir "/nyquist.csv"], "nyquist_data");

  t_step = (0:0.05:5)';
  y_step = 1 - exp(-t_step);
  step_data = [t_step(:), y_step(:)];
  save("-ascii", [outdir "/step.csv"], "step_data");

  disp("HELIOS Octave control systems ready");
end
