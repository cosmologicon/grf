import unittest
import grf

class GrfTest(unittest.TestCase):

	def testNodes(self):
		self.assertEqual([], grf.nodes([]))
		self.assertEqual(list("ABD"), grf.nodes(["AB", "AD"]))
		self.assertEqual(list("ABCD"), grf.nodes(["AB", "CD"]))

	def testIsConnected(self):
		self.assertTrue(grf.is_connected([]))
		self.assertTrue(grf.is_connected(["AB"]))
		self.assertFalse(grf.is_connected(["AB", "CD"]))
		self.assertFalse(grf.is_connected(["AB", "CD", "DF"]))
		self.assertTrue(grf.is_connected(["AB", "CD", "DB"]))

	def testHamiltonianPath(self):
		def checkHamiltonian(graph):
			path = grf.hamiltonian_path(graph)
			self.assertEqual(sorted(path), sorted(grf.nodes(graph)))
			graph = list(map(tuple, graph))
			for node0, node1 in zip(path, path[1:]):
				self.assertTrue((node0, node1) in graph or (node1, node0) in graph)
		checkHamiltonian(["AB"])
		checkHamiltonian("AB BC CD DE".split())
		checkHamiltonian("AB BC CD AD".split())
		self.assertFalse(grf.hamiltonian_path("AB CD".split()))
		self.assertFalse(grf.hamiltonian_path("AB AC AD".split()))

	def testExactCover(self):
		def checkExactCover(pieces):
			cover = grf.exact_cover(pieces)
			self.assertTrue(all(piece in pieces for piece in cover))
			nodes = [node for piece in cover for node in piece]
			all_nodes = set(node for piece in pieces for node in piece)
			self.assertEqual(sorted(nodes), sorted(all_nodes))
		checkExactCover("AB CD".split())
		checkExactCover("AB BC CD".split())
		checkExactCover([(1, 4, 7), (1, 4), (4, 5, 7), (3, 5, 6), (2, 3, 6, 7), (2, 7)])
		self.assertFalse(grf.exact_cover("AB BC".split()))

if __name__ == '__main__':
	unittest.main()

