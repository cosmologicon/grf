# Generate an exact cover problem. Each output line is a set of numbers 1-N.
# There is at least one combination of lines such that each number 1-N appears in exactly one line.


from __future__ import division, print_function
import random

N = 3000
s = 30   # approximate size of sets
m = 3000  # number of sets

nodes = list(range(N))

print(N)
nsolution = int(round(N / s))
subsets = [set() for _ in range(nsolution)]
for n in nodes:
	random.choice(subsets).add(n)

while len(subsets) < m:
	subsets.append([n for n in nodes if random.random() * N < s])

subsets = [tuple(sorted(subset)) for subset in subsets]
for subset in sorted(subsets):
	print(*subset)

