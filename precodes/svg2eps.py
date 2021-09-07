import os, glob, sys

path = sys.argv[1]
for svg in glob.glob(os.path.join(path, '*.svg')):
  eps_name = svg.split('.svg')[0]
  os.system('inkscape {} --export-eps={}.eps'.format(svg,eps_name))


