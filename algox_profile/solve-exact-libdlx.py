# Solve an exact cover problem from the command line using libdlx.
# Based on examples/count-solutions.py in the libdlx repository.
# See README.md for usage.

from __future__ import print_function
import sys, os

pydlx_dir = 'dlx'
sys.path.append(os.path.join(sys.path[0], pydlx_dir))

from pydlx.dlx_matrix import dlx_matrix
from pydlx.dlx_iterative_solver import dlx_iterative_solver

N = int(sys.stdin.readline().strip())
matrix = []
for line in sys.stdin:
	matrix.append([int(node) for node in line.split()])
num_nodes = sum(map(len, matrix))

with dlx_matrix(N, 0, num_nodes) as mat:
	for i, row in enumerate(matrix):
		mat.add_row(row, i)
	with dlx_iterative_solver(mat) as solver:
		while True:
			solution = solver.get_next_solution()
			if not solution:
				break
			print(*sorted(solution))


