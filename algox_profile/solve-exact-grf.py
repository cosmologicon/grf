# Solve exact cover from the command line using grf.
# See README.md for usage.

from __future__ import print_function
import sys
import grf

subsets = []
for line in sys.stdin:
	subsets.append([int(a) for a in line.split()])
subsets.pop(0)
subsets = dict(enumerate(subsets))

for solution in grf.exact_covers(subsets):
	print(*solution)

