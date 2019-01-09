# Solve the output of make-cover.py

from __future__ import print_function
import sys
import grf

subsets = []
for line in sys.stdin:
	subsets.append([int(a) for a in line.split()])

for subset in grf.exact_cover(subsets):
	print(*subset)
