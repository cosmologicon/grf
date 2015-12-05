from __future__ import print_function
import grf

# Knight's tour
adjacent = lambda s1, s2: sorted(abs(ord(x) - ord(y)) for x, y in zip(s1, s2)) == [1, 2]
squares = [f + r for f in "abcdefgh" for r in "12345678"]
edges = [(s1, s2) for s1 in squares for s2 in squares if s1 < s2 and adjacent(s1, s2)]
path = grf.hamiltonian_cycle(edges)
print(" ".join(path))
print()

# Removing the main diagonals
squares = [f + r for f, r in squares
	if ord(f) - ord(r) != ord("a") - ord("1") and ord(f) + ord(r) != ord("a") + ord("8")]
edges = [(s1, s2) for s1 in squares for s2 in squares if s1 < s2 and adjacent(s1, s2)]
path = grf.hamiltonian_cycle(edges)
print(" ".join(path))
print()

# Exact cover of a checkerboard using straight trominoes missing one square
pieces = [((x,y), (x,y+1), (x,y+2)) for x in range(8) for y in range(6)]
pieces += [((x,y), (x+1,y), (x+2,y)) for x in range(6) for y in range(8)]
pieces = [piece for piece in pieces if (2, 2) not in piece]
cover = grf.exact_cover(pieces)
pieceat = {pos: name for name, piece in zip("abcdefghijklmnopqrstu", cover) for pos in piece}
print("\n".join(" ".join(pieceat.get((x, y), ".") for x in range(8)) for y in range(8)))
print()

# Sudoku solver
grid = list("..24..59..6...1.47.....5......3.....3.45.6.2...6.7...44..7....8..9...1...2.....5.")
def get_constraints(row, column, digit):
	yield "f", row, column
	yield "r", row, digit
	yield "c", column, digit
	yield "b", int(row // 3), int(column // 3), digit
constraints = { (row * 9 + column, digit) : list(get_constraints(row, column, digit))
	for row in range(9) for column in range(9) for digit in "123456789" }
used_constraints = set().union(*[cons for (j, digit), cons in constraints.items() if grid[j] == digit])
available = list(filter(used_constraints.isdisjoint, constraints.values()))
solution = grf.exact_cover(available)
for (j, digit), cons in constraints.items():
	if cons in solution:
		grid[j] = digit
print("\n".join(" ".join(grid[row*9:row*9+9]) for row in range(9)))
print()

# 8 puzzle
start = 1, 5, 0, 8, 6, 7, 2, 4, 3
goal = tuple(range(9))
h = lambda state: sum(x != y for x, y in zip(state, range(8)))
def neighbors(state):
	i = state.index(8)
	y, x = divmod(i, 3)
	for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
		if 0 <= x + dx < 3 and 0 <= y + dy < 3:
			j = (y + dy) * 3 + (x + dx)
			s = list(state)
			s[i], s[j] = s[j], s[i]
			yield tuple(s)
for state in grf.astar_uniform(start, goal, neighbors, h):
	print(state)

