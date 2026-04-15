args = argv();

if numel(args) < 1
  disp("HELIOS Octave filter design ready");
else
  outdir = args{1};
  if ~exist(outdir, "dir")
    system(sprintf("mkdir -p '%s'", outdir));
  end

  w = linspace(0, pi, 256)';
  butter_mag = 1 ./ sqrt(1 + (w / (0.2 * pi)).^8);
  cheby_mag = 1 ./ sqrt(1 + 0.5 * cos(4 * acos(min(w / (0.2 * pi), 1))).^2);
  ellip_mag = 1 ./ sqrt(1 + 0.3 * (w / (0.2 * pi)).^4 + 0.8 * (w / (0.2 * pi)).^8);
  filter_data = [w(:), butter_mag(:), cheby_mag(:), ellip_mag(:)];
  save("-ascii", [outdir "/filter_designs.csv"], "filter_data");
  disp("HELIOS Octave filter design ready");
end
