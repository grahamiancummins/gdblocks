#!/usr/bin/env python -3
# encoding: utf-8

#Created by gic on Thu Jan 20 11:44:48 CST 2011

# Copyright (C) 2011 Graham I Cummins This program is free software; you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either version 2 of
# the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA 02111-1307 USA
#
"""
Tools for clustering or grouping. Includes a frontend for PyCluster

tags: 

DistMat(N); N,N-# of x; Symetric matrix where element i,j is the distance
between two entities i and j under some distance metric or dissimilarity 
measure. The entities referenced by the rows (or columns) are refered to as 
items 

Partition1(N); N-[ of i; Collection of integers indicating group membership, 
such that if self[i] == self[j], items i and j are in the same group.

Partition(M,N); M-[ of [ of i:sum([len(l) for l in self])==N ; alternate 
syntax for expressing a partition of N items into M groups. self[i] is a 
list of the items that are in group i.

other types:

Tree(N); Pycluster.tree instance. List-like. Contains nodes. Each node has 
	attributes left, right, distance. Distance is the distance between clusters.
	left and right identify them. If these are >=0, they are item indexes. If 
	they are <0 they reffer to other clusters in the tree, such that -X is 
	node self[X-1]. A Tree(N) is the result of clustering N objects. The length
	of the tree is N-1.

Node; An integer, which refers to a node in a Tree, obeying the sign convention
	mentioned in Tree. Unless otherwise specified Node may be either a node or a
	leaf. Node(+) is used to mean strictly a leaf, and Node(-) to mean strictly
	a node.

"""

from __future__ import print_function, unicode_literals
import numpy as np
from gicdat.util import infdiag
from mixmod import mmcall
try:
	from Pycluster import treecluster,kmedoids,kcluster, somcluster, Tree, Node
except:
	from gicdat.control import report
	report("Pycluster is not found. Several methods in the clust module will fail")
try:
	from zss import compare as zscmp
	from zss.test_tree import Node as ZSNode
except:
	from gicdat.control import report
	report("Zhang-shasha is not found. pct2zst and zssdist in clust will fail")
	


def dt_first(dm, t):
	'''
	dm: DistMat(N), t: x -> ids: Partition1(N)
	
	A naive grouping function which groups dm using a threshold, t. It depends
	on the ordering of dm, since it assigns each item to the same group as the
	first other item with distance <=t.
	
	'''
	centers=[]
	ids = []
	for id in range(dm.shape[0]):
		for c in centers:
			if dm[id, c] <=t:
				ids.append(c)
				break
		else:
			ids.append(id)
			centers.append(id)
	return ids	

def dt_hclust(dm, t):
	'''
	dm: DistMat(N), t: x -> ids: Partition(M, N)
	
	Partitions dm using hclustmean, which is a local implementation of mean-
	distance heirarchical clustering. Stops when the maximum distance between 
	clusters is <= t.
	
	'''
	memb = [[i] for i in range(dm.shape[0])]
	md = dm.max()
	while md>t:
		dm, memb =  hclustmean(dm, memb)
	return memb

DTHRESH = {'first':dt_first,
			'hclust':dt_hclust,
		}

def hclustmean(dm, memb):
	'''
	dm: DistMat(N), memb: Partition(N, L)  -> ndm: DistMat(N-1), 
		newmemb: Partition(N-1, L) 
	
	Performs one hierarchical clustering step using mean distance, based on the
	distance matrix dm and the membership array memb. Called internally by hca.
	
	'''
	dmi = infdiag(dm)
	closest = np.unravel_index(dmi.argmin(), dmi.shape)
	nm1 = len(memb[closest[0]])
	nm2 = len(memb[closest[1]])
	newc = memb[closest[0]]+memb[closest[1]]
	leaveout = [i for i in range(dm.shape[0]) if not i in closest]
	newmemb = [memb[i] for i in leaveout] + [newc]
	ndist = (dm[closest[0],:]*nm1 + dm[closest[1],:]*nm2)/(nm1+nm2)
	ndm = np.zeros((len(newmemb), len(newmemb)), dm.dtype)
	ndm[-1,-1] = ndist[closest[0]]
	ndist = ndist[leaveout]
	ndm[-1,:-1] = ndist
	ndm[:-1,-1] = ndist
	ndm[:-1,:-1] = dm[leaveout,:][:,leaveout]
	return (ndm, newmemb)


def hca(dists, memb, nclust):
	'''
	dists: DistMat(N), memb: Partition(N, L), nclust: i  
		-> (DistMat(nclust), Partition(nclust, L)) 
	
	Construct nclust clusters using agglomerative hierarchical clustering with
	mean distance. The inputs dists and memb specify an existing clustering. If 
	memb is false, it defaults to [ [i] for i in range(dists.shape[0])] (which 
	correponds to no initial clustering.
	
	This implementation is slower than dtree (which uses pycluster)
	'''
	if not memb:
		memb = [ [i] for i in range(dists.shape[0])]
	while len(memb)>nclust:
		dists, memb = hclustmean(dists, memb)
	return dists, memb

def mixmodpartition(data, k,model="Gaussian_pk_Lk_Ck", reps = 1):
	"""Wraps mixmod.mmcal(data, [k], model, False, reps), and returns the 
	resulting partition"""
	d = mmcall(data, [k], model, False, reps)
	return d['partition']

#dtree(dists, **cargs)		
#clust.cuttree(t, nclust)
#clust.mediods(dists, nclust, **cargs)
#clust.vtree(evts, **cargs)
#clust.kmeans(evts, nclust, **cargs)

#method: s, m, c, a    single-link, max, centroid, average 

def dtree(dm, method='a'):
	"""
	dm: DistMat(N), method: s ('a') -> Tree(N)
	
	calls PyCluster.treecluster(method=method, distancematrix=dm), except that 
	dm is copied, to protect from PyClusters INSANE behaviour of modifying dm 
	in place. 
	
	"""
	#warning: pycluster writes the dm by side effect!!!!!
	dm = dm.copy()
	return treecluster(method=str(method), distancematrix=dm)

def vtree(dat, dist='e', method='a'):
	"""
	dat: N,M-#, dist: s ('e'), method: s ('a') -> Tree(N)
	
	calls PyCluster.treecluster(dat, dist=dist, method=method),
	
	This require that dat contain data smaples in rows, and that the distance
	function is appropriate to use on the samples (e.g. the default dist='e' is 
	the euclidean distance, and all PyCluster dist options are vector space 
	distances).
	
	"""
	#implicit transpose=0, aka samples are rows, and there is no mask or weight
	return treecluster(dat, dist=dist, method=str(method))

# clustering here seems to work, but it calls pycluster.distancematrix, which 
# normalizes in a strange way. In particular, selecting euclidean distance actually 
# gets you the square of the euclidean distance divided by the number of dimensions

#pycluster measures
#vspace measures:
#k means 
#data nclusters, mask, weight, transpose, npass, method, dist, initialid		
#mask weight transpose can probably be ommited. 

#dist mat measures:
#k-mediods
#distance, nclusters, npass, initialid
#tree
#data, mask, weight, transpose, method, dist, distancematrix

def tree2dot(tree, names=None):
	'''
	tree: Tree(N), names: [ of s (*) -> s
	
	(*) if False, defaults to ['n0', 'n1', ... 'nN']
	
	Makes a representation of a tree object in the graphviz dot language. 
	
	'''
	from matplotlib.cm import get_cmap
	spectral = get_cmap('spectral')
	if not names:
		names = ["n%i" % (i, ) for i in range(len(tree)+1)]
	gv = ['digraph G {']
	dists = [n.distance for n in tree]
	dmax = max(dists)
	dmin = min(dists)
	def _nn(j):
		if j>=0:
			return names[j]
		else:
			return "c%i" % (-1*j,)
	def _d2c(d):
		if dmin==dmax:
			return "#000000"
		d1 = (d - dmin)/(dmax-dmin)
		c = spectral(d1, bytes=True)[:3]
		s = "#%02x%02x%02x" % c
		return s
	for i in range(len(tree)-1, -1, -1):
		nid = "c%i" % (i+1,)
		lid = _nn(tree[i].left)
		rid = _nn(tree[i].right)
		dist = _d2c(tree[i].distance)
		gv.append('"%s" -> "%s" [penwidth=2, color="%s"];' % (nid, rid, dist))
		gv.append('"%s" -> "%s" [penwidth=2, color="%s"];' % (nid, lid, dist))
	gv.append('}')
	return "\n".join(gv)


def tree2tup(t):
	"""Converts a Pycluster Tree to a list of tuples (which can be stored in a
	gd.Doc, repr/eval group, etc)"""
	return tuple( [(n.left, n.right, n.distance) for n in t] )

def tup2tree(t):
	"""Restores a PyCluster Tree from the output of tree2tup"""
	return Tree([Node(*n) for n in t])
	


def treecontains(tree, node, leaf):
	'''
	tree: Tree(N), node: Node, leaf: Node -> bool
	
	True if node in tree contains leaf (recursively), False otherwise. Both node
	and leaf may be negative (node references) or non-negative (leaf
	references). If node is non-negative this is the recursion edge-case, and
	just tests node==leaf.
	
	'''
	if node>=0:
		return node==leaf
	else:
		n = tree[-node-1]
		return (treecontains(tree, n.left, leaf) 
		        or treecontains(tree, n.right, leaf))

def treebranch(tree, node):
	"""
	tree: Tree, node: Node -> [ of i
	
	returns all the elements that are children of node in tree. If node is 
	non-negative, this is the list [node], of corse. 
	"""
	if node>=0:
		return [node]
	else:
		n = tree[-node-1]
		return treebranch(tree, n.left) + treebranch(tree, n.right)

def treelevel(tree, node, leaf):
	'''
	tree: Tree(N), node: Node, leaf: Node -> i
	
	If treecontains(tree, node, leaf), the return value is the number of levels
	down from node that leaf is found. Otherwise, it is -1
	
	'''
	if node>=0:
		if node==leaf:
			return 0
		else:			
			return -1
	else:
		n = tree[-node-1]
		d = treelevel(tree, n.left, leaf)
		if d == -1:
			d = treelevel(tree, n.right, leaf)
			if d == -1:
				return -1
		return d+1
	
def treeparent(tree, node):
	"""
	tree: Tree, node: Node -> None | Node(+)
	
	returns the node that is the direct parent of the input node. The input 
	may be negative or non-negative. The return value will be negative (since 
	a leaf is never a parent). If the specified node is the root (there is no 
	parent) the return is None.
	
	"""
	for i, n in enumerate(tree):
		if tree[i].left == node or tree[i].right == node:
			return -i-1
	return None

def treedist_lev(tree, n1, n2):
	"""
	tree: Tree, n1: Node, n2: Node -> i | inf

	Returns the number of levels up the tree you need to climb from n1 to reach
	a node that contains n2. Return is np.inf if one of the nodes is not in tree.
	
	Note that this is not a symettric function, since if one node is close to 
	the root, all level distances will be very small (and also this is non-
	intuitive, since such a node is usually an outlier that was clustered last).
	"""
	d = 0
	while not treecontains(tree, n1, n2):
		d+=1
		n1 = treeparent(tree, n1)
		if n1==None:
			return np.inf
	return d

def treedist_levs(tree, n1, n2):
	"""
	tree: Tree, n1: Node, n2: Node -> i | inf
	
	Average of treedist_lev(tree, n1, n2) and treedist_lev(tree, n2, n1)
	(thus symettric) 
	
	"""
	return (treedist_lev(tree, n1, n2)+treedist_lev(tree, n2, n1))/2.0
	
def treedist_links(tree, n1, n2):
	"""
	tree: Tree, n1: Node, n2: Node -> i | inf
	
	Number of edges that need to be traversed to get from n1 to n2. This should 
	be twice treedist_levs(tree, n1, n2), and somewhat faster to calculate. 
	"""
	d = 0
	tl = treelevel(tree, n1, n2)
	while tl==-1:
		d+=1
		n1 = treeparent(tree, n1)
		if n1==None:
			return np.inf
		tl = treelevel(tree, n1, n2)
	return d+tl

def pairs(l):
	"""
	l: [ of ? -> [ of 2-[ of ?
	
	returns a list of pairs containing all unique pairs of items in list l.
	This is a combination not a permutation, and also it assumes an item can't
	be paired with itself. Length of the return, if the input length is N, 
	is thus N!/( 2*(N-2)! )
	"""
	ret = []
	for i in range(1, len(l)):
		for j in range(i):
			ret.append( (l[j], l[i]) )
	return ret

def cmptrees(t1, t2):
	"""
	t1:Tree(N), t2:Tree(N) ->op of M,M-# of i
	
	Compares two trees using treedist_links. This assumes that the trees
	connect the same nodes (e.g. that the semantics of the item indexes in the 
	trees are the same), but only checks for the syntax (that the trees have the 
	same length, and thus that the set of actual indexes is the same). 
	
	The function checks all pairs of nodes, and compares their distance (in the
	links sense) in t2, and t2. The matrix op is a 2D histogram, such that
	op[i,j] specifies the number of pairs that had distance i+2 in t1, and
	distance j+2 in t2. The +2 is used to compress the matrix, since distances
	of 0 or trivial (node identity), and distance 1 is imposible under
	treedist_links, so 2 is the smallest distance of interest, and is listed in
	the first row/column. The size of op depends on how balanced the trees are.
	It is large enough to contain the largest distance that occurs (in either
	tree), which is on the range (log2(N), N+1).
	
	Identical trees will have only diagonal elements in op, so 
	op.sum - np.diag(op).sum() gives a measure of dis-similarity between the 
	trees.
	
	"""
	if len(t1)!=len(t2):
		raise ValueError("Can't compare trees of different sizes")
	n = len(t1)
	md = 0
	op = np.zeros((n+2, n+2))
	coords = []
	ps = pairs(range(n))
	for p in ps:
		d1 = treedist_links(t1, p[0], p[1])
		d2 = treedist_links(t2, p[0], p[1])
		op[int(d1), int(d2)]+=1
		md = max(md, max(d1, d2))
	return op[2:md+1,2:md+1]
	
def branchlengths(t, n):
	"""
	t: Tree, n: i -> (i, i)
	
	returns a tuple (min, max) of the lengths of the various subtrees of 
	node n in tree t. If n is a leaf (n>=0), this is (0, 0). Otherwise, min is 
	the smallest number of edges required to get from n to a leaf, and max is 
	the largest number possible.
	
	"""
	if n>=0:
		return (0, 0)
	else:
		nid = -n - 1
		bll = branchlengths(t, t[nid].left)
		blr = branchlengths(t, t[nid].right)
		return (min(bll[0], blr[0])+1, max(bll[1], blr[1])+1)

def uplength(t, n, camefrom=None):
	"""
	As branch lengths, but considers paths that go upward as well 
	"""
	if n>=0:
		return (0, 0)
	else:
		nid = -n - 1
		milen = len(t)+2
		malen = 0
		pn = treeparent(t, n)
		if pn and pn != camefrom:
			mi, ma = uplength(t, pn, n)
			milen= min(mi, milen)
			malen = max(ma, malen)
		if t[nid].left!=camefrom:
			mi, ma = uplength(t, t[nid].left, n)
			milen= min(mi, milen)
			malen = max(ma, malen)
		if t[nid].right!=camefrom:
			mi, ma = uplength(t, t[nid].right, n)
			milen= min(mi, milen)
			malen = max(ma, malen)
		return (milen+1, malen+1)

def _root(t):
	nodes = set([-n-1 for n in range(len(t))])
	for n in t:
		nodes-= set([n.left, n.right])
	if len(nodes)==1:
		return nodes.pop()
	return nodes
	
def treesort(t, lf=False, n=None):
	"""
	t: Tree, lf: t->s: [ of i
	
	s is an ordering of the leafs in t (e.g. a list containing one copy each of 
	every non-negative integer in t), which depends on the structure of t
	
	lf specifies whether to sort "longest first".
	
	"""
	if n is None:
		n = _root(t)
		if type(n)==set:
			raise ValueError('use treesort_rootless for this tree')
	if n >=0:
		return [n]
	nid = -n -1 
	nl = branchlengths(t, t[nid].left)
	nr = branchlengths(t, t[nid].right)
	lts = treesort(t, lf, t[nid].left)
	rts = treesort(t, lf, t[nid].right)
	if lf:
		if nl[1]<nr[1]:
			return rts+lts
		else:
			return lts+rts
	else:
		if nl[0]>nr[0]:
			return rts+lts
		else:
			return lts+rts

def tclusters(t):
	"""
	t: Tree -> [ of [ of i
	
	Returns a partition list for the tree t
	"""
	return [treebranch(t, -i - 1) for i in range(len(t))]

def compat(c1 , c2):
	"""
	c1:[ of i, c2:[ of i -> t
	
	are cluters c1 and c2 compatible (aka identical, disjoint, or nested)
	"""
	c1 = set(c1)
	c2 = set(c2)
	if c1.isdisjoint(c2):
		return True
	elif c1.issubset(c2):
		return True
	elif c2.issubset(c1):
		return True
	return False

def contained_in(ft, ft2):
	inft2 = set()
	for f in ft2:
		inft2 = inft2.union(f)
	new = []
	for f in ft:
		new.append(sorted(list(inft2.intersection(f))))
	return new

def equal_clust_frac(ft1, ft2):
	n = 0
	for c in ft1:
		s1 = set(c)
		for c2 in ft2:
			if set(c2) == s1:
				n+=1
				break
	return float(n)/len(ft1)

def clustcounts(trees):
	atc = {}
	for t in trees:
		ttc = [tuple(sorted(s)) for s in tclusters(t) if len(s) <= len(t)]
		for c in ttc:
			if c in atc:
				atc[c]+=1
			else:
				atc[c] = 1
	return atc

def _tupsort(t1, t2):
	return cmp(len(t1), len(t2)) or cmp(t1, t2)
	
def mrtree(trees):
	atc = clustcounts(trees)
	ft = [k for k in atc if atc[k] > len(trees)/2.0]
	nr = [float(atc[k])/len(trees) for k in ft]
	return (ft, nr)

def _lchild(t, z):
	ii = -1
	ll = 0
	for i, tt in enumerate(z):
		if t.issuperset(tt) and len(t) > len(tt):
			l = len(tt)
			if l>ll:
				ii = i
				ll = l
	return ii
	
def buildtree(ft, n, nr=None):
	nn = [[set(range(n)), None, None, None]]
	if nr!=None:
		nn[-1][-1] = 1.0
	active = [0]
	while active:
		ni = active.pop(0)
		lc = _lchild(nn[ni][0], ft)
		if lc == -1:
			pass
			#print('Warning: tree incomplete')
		else:
			if nr!=None:
				prob = nr[lc]
			else:
				prob = None
			lset = set(ft[lc])
			rset = nn[ni][0].difference(lset)
			for i, s in enumerate([lset, rset]):
				if len(s) == 1:
					nn[ni][i+1] = s.pop()
				else:
					id = -len(nn)-1
					nn[ni][i+1] = id	
					if len(s) == 2:
						st = tuple(s)
						nn.append([s, st[0], st[1], prob])
					else:
						active.append(len(nn))
						nn.append([s, None, None, prob])
	nodeids = {}
	return nn

	

def ctree2dot(ft, names, nr=None):
	tree = buildtree(ft, len(names), nr)
	gv = ['digraph G {']
	for i in range(len(tree)):
		node = tree[i]
		nid = "c%i" % (i+1,)
		c = tuple(sorted(node[0]))
		if node[-1] != None:
			gv.append('%s [label="%s ( %.2f )"];' % (nid, nid, node[-1]))
		if node[1] is None:
			for k in c:
				gv.append('"%s" -> "%s";' % (nid, names[k]))
		else:
			for v in [1,2]:
				if node[v]<0:
					tn = "c%i" % (-1*node[v],)
				else:
					tn = names[node[v]]		
				gv.append('"%s" -> "%s";' % (nid, tn))
	gv.append("}")
	return "\n".join(gv)

def clustcompatscore(ct1, ct2):
	"""
	each ct is (ft, nr)
	
	"""
	s = 0
	for i1, c1 in enumerate(ct1[0]):
		for i2, c2 in enumerate(ct2[0]):
			if compat(c1, c2):
				s+=ct1[1][i1]+ct2[1][i2]
	return s

#zscmp ZSNode from zss

def pct2zst(t, names):
	"""
	t: Tree -> ZSNode
	
	Convert a pycluster Tree into the tree type used by Zhang-shasha. The 
	return type is a zss.test_tree.Node, containing the root of the tree.
	"""
	r = t[-_root(t)-1]
	rn = ZSNode('c')
	def addnode(t, i):
		if i>=0:
			n = ZSNode(names[i])
		else:
			n = ZSNode('c')
			tn = t[-i-1]
			for k in [tn.left, tn.right]:
				n.addkid(addnode(t, k) )
		return n
	return addnode(t, _root(t))
	
def zst2dot(zt):
	"""
	zt : ZSNode -> s
	
	Return a dot language representation of the ZSNode zt
	"""
	gv = ['digraph G {']
	used = []
	def relabel(n):
		if n.label in used:
			i = 2
			while "%s%i" % (n.label, i) in used:
				i+=1
			n.label = "%s%i" % (n.label, i)
		used.append(n.label)
	def addnode(n):
		relabel(n)
		for k in n.children:
			addnode(k)
			gv.append('"%s" -> "%s";' % (n.label, k.label))
	addnode(zt)
	gv.append("}")
	return "\n".join(gv)	

def ft_nr2pt(ftt, names):
	nodes = {}
	for i, n in enumerate(names):
		nodes[i] = ZSNode(n)
	t = buildtree(ftt[0], len(names), ftt[1])
	for i, n in enumerate(t):
		nid = -i-1
		if not nid in nodes:
			nodes[nid] = ZSNode('c')
		if n[1] != None:
			s = [n[1], n[2]]
		else:
			s = list(n[0])
		for v in s:
			if not v in nodes:
				nodes[v] = ZSNode('c')
			nodes[nid].addkid(nodes[v])
	return nodes[-1]

def t2zst(t, names):
	"""
	distributer function that checks the type of tree structure t and calls the 
	appropriate *2zst function to produce a Zhang-shasha compatible tree.
	
	"""
	if type(t) == Tree:
		return pct2zst(t, names)
	else:
		return ft_nr2pt(t, names)
	
	
	
def zssdist(t1, t2, names):
	"""
	t1: Tree, t2:Tree -> x
	
	Return the edit distance, as computed by zss.compare, between t2zst-compatible trees t1, t2
	"""
	t1, t2 = map(lambda x:t2zst(x, names), [t1, t2])
	return zscmp.distance(t1, t2)