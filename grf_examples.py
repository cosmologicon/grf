from __future__ import print_function
from collections import defaultdict
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
available = {name: cons for name, cons in constraints.items() if used_constraints.isdisjoint(cons)}
solution = grf.exact_cover(available)
for j, digit in solution:
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
print()

# Samurai sudoku

grid = list(map(list, [
	"6...983.7   4..15...6",
	"...6.1...   ..96.3...",
	"...2....4   ...8.....",
	".6.1.78..   ..17.859.",
	"7...4..6.   9..51...3",
	"18593....   5....9.67",
	".2.3......52.....1..8",
	".......4...7.6..8....",
	"9...84...9.4......4.9",
	"      637...5..      ",
	"      4.......9      ",
	"      ..8...742      ",
	"1.8......2.1...21...3",
	"....3..1.7...2.......",
	"9..7.....84......4.6.",
	"56.4....9   ....25734",
	"8...93..6   .8..7...1",
	".198.52..   ..21.9.8.",
	".....1...   1....2...",
	"...6.74..   ...6.3...",
	"2...49..7   3.845...2",
]))
def get_constraints(row, column, digit):
	yield "f", row, column
	yield "r", row, digit
	yield "c", column, digit
	yield "b", int(row // 3), int(column // 3), digit
offsets = (0, 0), (12, 0), (6, 6), (0, 12), (12, 12)
constraints = defaultdict(list)
for row in range(9):
	for column in range(9):
		for digit in "123456789":
			for constraint in get_constraints(row, column, digit):
				for jgrid, (offx, offy) in enumerate(offsets):
					constraints[(column + offx, row + offy, digit)].append((jgrid, constraint))
used_constraints = set().union(*[cons for (x, y, digit), cons in constraints.items() if grid[y][x] == digit])
available = {name: cons for name, cons in constraints.items() if used_constraints.isdisjoint(cons)}
solution = grf.exact_cover(available)
for x, y, digit in solution:
	grid[y][x] = digit
print("\n".join(" ".join(row) for row in grid))
print()


