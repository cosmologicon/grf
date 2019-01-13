# Generate an exact cover problem. Each output line is a set of numbers 1-N.
# There is at least one combination of lines such that each number 1-N appears in exactly one line.

# sudo apt install ghc
# https://hackage.haskell.org/package/exact-cover




from __future__ import division, print_function
import random

N = 10000
s = 100   # approximate size of sets
m = 3000  # number of sets


nsolution = int(round(N / s))
subsets = [set() for _ in range(nsolution)]
for n in range(1, N+1):
	random.choice(subsets).add(n)

while len(subsets) < m:
	subsets.append([n for n in range(1, N+1) if random.random() * N < s])

subsets = [tuple(sorted(subset)) for subset in subsets]
for subset in sorted(subsets):
	print(*subset)

