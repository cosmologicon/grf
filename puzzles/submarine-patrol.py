# http://web.mit.edu/puzzle/www/2018/full/puzzle/submarine_patrol.html

import grf

N = 7
shipcounts = { 1: 4, 2: 9, 3: 9, 4: 4 }
columncounts = [
	[4, None, 2, 1, 2, None, 2],
	[2, 0, 2, 0, None, 0, None],
	[4, None, None, 1, None, 2, 2],
	[None, None, 1, 5, None, 0, None],
	[2, 0, None, None, None, 3, 1],
	[1, None, 3, 0, 0, 0, 0],
	[1, None, 1, 1, 2, 3, None],
]
rowcounts = [
	[0, 4, 0, 3, 1, 4, 1],
	[None, None, None, 3, 0, 1, 0],
	[2, 1, 0, 2, None, None, None],
	[1, 0, 1, None, None, 1, 2],
	[1, 1, 0, 2, 1, None, None],
	[3, None, 0, 0, 0, None, None],
	[2, None, 4, 0, None, 3, 1],
]

ranges = {}
for ship_size, count in shipcounts.items():
	ranges[("s", ship_size)] = count
for z, columncount in enumerate(columncounts):
	for x, count in enumerate(columncount):
		ranges[("c", x, z)] = (0, 7) if count is None else count
for z, rowcount in enumerate(rowcounts):
	for y, count in enumerate(rowcount):
		ranges[("r", y, z)] = (0, 7) if count is None else count
for x in range(N):
	for y in range(N):
		for z in range(N):
			ranges[("o", x, y, z)] = 0, 1

pieces = {}
def covers_of(cells):
	covers = set((x + dx, y + dy, z + dz) for x, y, z in cells for dx in (0, 1) for dy in (0, 1) for dz in (0, 1))
	covers = [("o", x, y, z) for x, y, z in covers if x < N and y < N and z < N]
	covers += [("s", len(cells))]
	for x, y, z in cells:
		covers += [("c", x, z), ("r", y, z)]
	return covers

shipstarts = [(x, y, z) for x in range(N) for y in range(N) for z in range(N)]
for ship_size in shipcounts:
	shipdirs = [(0, 0, 1), (0, 1, 0), (1, 0, 0)] if ship_size > 1 else [(0, 0, 0)]
	for x, y, z in shipstarts:
		for dx, dy, dz in shipdirs:
			cells = tuple((x + j * dx, y + j * dy, z + j * dz) for j in range(ship_size))
			if all(x1 < N and y1 < N and z1 < N for x1, y1, z1 in cells):
				pieces[cells] = covers_of(cells)

solution = grf.multi_cover(pieces, node_ranges = ranges)
grid = { (x, y, z): "." for x in range(N) for y in range(N) for z in range(N) }
for piece in solution:
	for x, y, z in piece:
		grid[(x, y, z)] = str(len(piece))
for z in range(N):
	for y in range(N):
		print(*(grid[(x, y, z)] for x in range(N)))
	print()




