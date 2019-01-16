# Generate an exact cover problem. Each output line is a set of numbers 1-N.
# There is at least one combination of lines such that each number 1-N appears in exactly one line.


from __future__ import division, print_function
import random, sys

N = 40000
s = 200   # approximate size of sets
m = 4000  # number of sets

if True:
	N, s, m = [int(arg) for arg in sys.argv[1:]]

nodes = list(range(N))

print(N)
nsolution = int(round(N / s))
subsets = [set() for _ in range(nsolution)]
for n in nodes:
	random.choice(subsets).add(n)

while len(subsets) < m:
	subsets.append([n for n in nodes if random.random() * N < s])

subsets = [tuple(sorted(subset)) for subset in subsets if subset]
for subset in sorted(subsets):
	print(*subset)

