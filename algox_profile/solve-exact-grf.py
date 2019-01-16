# Solve exact cover from the command line using grf.
# See README.md for usage.

from __future__ import print_function
import sys
import grf

lines = [[int(a) for a in line.split()] for line in sys.stdin]
N, = lines.pop(0)
sys.setrecursionlimit(N)
subsets = dict(enumerate(lines))
for solution in grf.exact_covers(subsets):
	print(*solution)

