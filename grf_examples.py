import grf

# Knight's tour
adjacent = lambda s1, s2: sorted(abs(ord(x) - ord(y)) for x, y in zip(s1, s2)) == [1, 2]
squares = [f + r for f in "abcdefgh" for r in "12345678"]
edges = [(s1, s2) for s1 in squares for s2 in squares if s1 < s2 and adjacent(s1, s2)]
path = grf.hamiltonian_cycle(edges)
print(" ".join(path))

# Removing the main diagonals
squares = [f + r for f, r in squares
	if ord(f) - ord(r) != ord("a") - ord("1") and ord(f) + ord(r) != ord("a") + ord("8")]
edges = [(s1, s2) for s1 in squares for s2 in squares if s1 < s2 and adjacent(s1, s2)]
path = grf.hamiltonian_cycle(edges)
print(" ".join(path))

