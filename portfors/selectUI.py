#!/usr/bin/env python -3
# encoding: utf-8

#Created by Graham Cummins on Jun 7, 2011

# Copyright (C) 2011 Graham I Cummins
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later 
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA 02111-1307 USA
#
"""
Functions for highlevel project control on the stimulus selectivity project.

This now also includes several functions intended for comparison and
visualization of trees, which derive from clustering of inter-class distance
matrixes.


tags:

CellExp from timingUI
Tree from clust
Roc from classify

(*) Clearly, all extension packages should define a single tags.py with all 
the tags used in the package. It avoids deeply nested nomenclature, and avoids
having to declare tag sharing, which is very common. 

(**) By convention I'm using "t" for boolean. The API in gicdat search 
includes boolean as "i" (integer), so the full spec would be 
i::self in [0,1]. This is fine, as far as the range of allowed data are 
concerned, but too slow, in the context of function signatures, which 
discriminate boolean flags from integers all the time. 

CellDoc: {KeySet('cell'):CellExp}
	
	This document contains keys such as "cell50", refferencing CellExp
	sub-documents. The sub documents contain compatible stimuli, meaning that 
	the same set of stimuli is used in each CellExp, and that the same stimulus
	id (integer) is always used to reffer to the same actual stimuls (waveform). 
	The keys are typically discontinuous (e.g. cell32 and cell35 may be included,
	but this doesn't imply that there is a cell33). 
	
	This module assumes that all CellExp in a CellDoc define conditions "cond1"
	and "cond2", and if any others are defined they will usually be ignored.
	
	This tag is the return type of timingUI.celldoc

CondName(d): Key(d, IntIO) (***)

	A clear type pattern which should be added to the spec is Key(d, p?).
	Key(d) are items in the set of keys of dictionary d. If the second 
	parameter "p" is given, then a Key(d, p) is a key that is in d, such that 
	d[p] matches pattern p. In this case, a CondName(d) is a key into Doc d 
	that reffers to an IntIO subdocument (eg. "cond1" or "cond2" in most of our
	experiments)


StimClass(S): Abstract

	The set of all responses to a particular (enumerated) stimulus condition.
	In the context of an IntIO, this could be represented as a partition or 
	a list of integers, e.g.
	[i for i in range(len(io['stims']) if  io['stims'][i] == S ]
	In this module, we use Stimulus S to be a string, which is the human-readable
	name of the stimulus file, rather than the enumeration index.

DistMode: s (****)

	Type "s" here is poor. It is passed as the argument "mode" to dist.dist but
	there is no type signature for that argument currently provided in module
	dist. It is an enumeration, since it needs to key dictionaries in module
	dist, but it isn't exactly a Key(dist.DMATS), or similar, because the
	function dist.dist parses the string if it contains an "_". It is exactly:
	Key(dist.DMATS) | Key(dist.VDISTS)+"_"+Key(dist.VREPS) I'm not sure how to
	conscisly represent that idea yet. Currently, the exhaustive enumeration is:
	vd, vdps, vdpe, ed_raw, ed_bin, ed_ist, ed_isi, ed_irate, cb_bin, cb_ist
	cb_isi, cb_irate of these vd, vdps, ed_ist, and ed_bin seem to be useful.
	vdpe (the expected poisson distance correction to the victor distance) would
	be useful, but the implementation in dist isn't complete yet.

DistQ: x

	The precision parameter used by some distance measures. Interprettation 
	depends on a DistMode. For example, in binning measures, this is the 
	bin width in microseconds. In Victor measures, this is 2.0/Vq, where Vq
	is Victor's "q" parameter - the cost of shifting an event by 1 microsecond.
	This also has the interprettation of being the longest distance by which a
	spike will ever be translated during the Victor transformation sequence. 

ICD: (dm of DistMat(N), names of N-[ of s)

	dm[i,j] is the expected distance between StimClass(s1) and Stimclass(s2)
	under some distance measure (a fully dependent version of the type would be
	ICDD(d: IntIO, m: DistMode, p: DistQ). The expected distance is the average
	of all pairwise distances between r_i, r_j where r_i is in StimClass(s1) and
	r_j in StimClass(s2). If s1==s2 the 0 distances between a response and
	itself are not considered in the average.
	
DirName: s
	Specifies the name of a directory in the file system. It will always be
	passed through os.path.expanduser beofore being used, meaning that "~" can
	be used to represent the current user's home directory. 

"""

import matplotlib.pyplot as plt
import gicdat.doc as gd
import gicdat.io as gio
import numpy as np
from timingUI import load, celldoc, preproc, GOOD, showcell
from timingUI import rfromstim
from gicdat.enc import flat
import dist, vis, clust, os
from classify import roc
import tests

def stimnames(d):
	astim = set()
	for k in d:
		if k.startswith('cond'):
			astim = astim.union(set(d[k]['stims']))
	mas = max(tuple(astim))
	snames = []
	sc=d['stimclasses'] or {}
	for i in range(mas+1):
		if "stim%i" % i in sc:
			snames.append(sc["stim%i" % i]['file'].split('.')[0])
		else:
			snames.append("Unknown Stim")
	return snames

def maxspikes(cells):
	"""
	cells: CellDoc -> i
	
	return the largest number of spikes that result from any stimulus 
	presentation (across all cells and conditions 1, 2) in the CellDoc.
	"""
	cd = gd.Doc()
	mns = 0
	for cn in cells:
		for cond in ['cond1', 'cond2']:
			evts = cells[cn][cond]['evts']
			mls = max([len(e) for e in evts])
			mns = max(mns, mls)
	return mns

def snhist(cells):
	"""
	cells:CellDoc -> hists: {cond1:N-#, cond2:N-#, cond1n:i, cond2n:i}
	
	Returns a dictionary of histograms. The keys cond# conatain arrays A such 
	that A[i] is the number of presentations (of any stimulus, in any cell, in
	condition cond#) that resulted in i spikes. cond#n is the total number of 
	presentations (of any stimulus in any cell in condition cond#)
	
	"""
	mns = maxspikes(cells)
	hists = {'cond1':np.zeros(mns+1), 'cond2':np.zeros(mns+1),
	         'cond1n':0, 'cond2n':0}
	for cn in cells:
		for cond in ['cond1', 'cond2']:
			evts = cells[cn][cond]['evts']
			for e in evts:
				ns = len(e)
				hists[cond][ns]+=1
				hists[cond+'n']+=1
	return hists

def rrps(cells):
	"""
	cells: CellDoc -> (cns:N-[ of s, sns:N-[ of s, 
						hists:{cond1 cond2:N,N-# of x})
	
	cns is a list of cell names, sns is a list of stimulus names, hists is 
	a dictionary of arrays such that hists[c][i,j] is the fraction of total 
	responses of cell cns[i] in condition c which occured in respons to stimulus
	sns[j].
	
	"""
	cns = cells.keys(0, 'cell', sort=True)
	stims = stimnames(cells[cns[0]])
	hists = {'cond1':np.zeros((len(cns), len(stims))),
	         'cond2':np.zeros((len(cns), len(stims)))}
	for i, cn in enumerate(cns):
		for cond in ['cond1', 'cond2']:
			nts = len(flat(cells[cn][cond]['evts']))
			if not nts:
				continue
			for sno, sname in enumerate(stims):
				evts = rfromstim(cells[cn], cond, sno)
				if evts:	
					rr = float(len(flat(evts)))/float(nts)
				else:
					rr = 0
				hists[cond][i, sno] = rr
			#hists[cond][i,:] = np.argsort(hists[cond][i,:])
	return (cns, stims, hists)

def show_rrps(cells):
	"""
	cells: CellDoc -> None (draws in mpl figure 1)
	
	calls rrps(cells), and displays the result, using proportions of response 
	spikes (that is, each row in the figure has the total height of all 
	bars in the same condition sum to 1). Blue is condition 1, Red, condition 2
	
	"""
	cns, sns, h = rrps(cells)
	f = plt.figure(1)
	plt.clf()
	x = np.arange(len(sns))
	mr = max(h['cond1'].max(), h['cond2'].max())
	for i in range(len(cns)):
		c1d = h['cond1'][i,:]
		c2d = h['cond2'][i,:]
		lmr = max(c1d.max(), c2d.max())
		if lmr!=0:
			c1d = .8*c1d/lmr
			c2d = .8*c2d/lmr
		plt.bar(x, c1d, color='b', bottom=i, width=.9)
		plt.bar(x, c2d, color='r', bottom=i, width=.6)
	plt.yticks(np.arange(len(cns)), cns)
	plt.xticks(np.arange(len(sns))+.5, sns)
	plt.ylim([-.2, len(cns)+.2])
	f.canvas.draw()
	
def show_snhist(cells):
	"""
	cells: CellDoc -> None (draws in mpl figure 1)
	
	Calls snhist(cells), and draws a bar chart of the results. This is the 
	histogram of response rates for experimental conditions 1 (blue) and 2 (red)
	
	"""
	h = snhist(cells)
	f = plt.figure(1)
	plt.clf()
	x = np.arange(len(h['cond1']))
	plt.bar(x, h['cond1']/float(h['cond1n']), color='b', label='control')
	plt.bar(x, h['cond2']/float(h['cond2n']), color='r',
	        width=.5, label='Bic/Strych')
	plt.xlabel('Number of spikes')
	plt.ylabel('Fraction of responses')
	plt.legend(loc='best')
	plt.xlim([0, 30])
	plt.title('Response rates with and without inhibition')
	f.canvas.draw()

def allrasters(cells = GOOD, save="~/Dropbox/wsu/cp_ic_rasters"):
	"""
	cells: [ of i, save: DirName -> None (writes many files, draws in MPL 
		figure 1)
		
	Differs from the function of the same name in timingUI in that it calls 
	celldoc(cells) to load data, which means that only the common stimuli are 
	included 
	
	"""
	save = os.path.expanduser(save)
	d = celldoc(cells)
	for cn in d:
		showcell(d[cn], save=os.path.join(save, "%s.png" % cn))

def dist_discrim_incond(d, mode='vdps', q=6000, nstim = None):
	"""
	d: IntIO, mode: DistMode, q: x -> 
		ret: ICD(d[cond], mode, q)
	
	norm is a flag. If True, the inter-class distances are normalized by the 
	expected inclass distance: 
	
	"""
	dm = dist.dist(d['evts'], mode, q)
	dc = dist.classDM(dm, d['stims'], nstim)
	dca = np.zeros( (len(dc), len(dc)))
	for i in range(len(dc)):
		for j in range(len(dc[i])):
			dca[i, j] = np.mean(dc[i][j])
			if i!=j:
				dca[j,i] = dca[i,j]
	return dca

def _draw_icd(a, labs, vmin=None, vmax=None):
	"""Internal, used by show_icd and friends"""
	plt.imshow(a[::-1,:], interpolation='nearest', vmin=vmin, vmax=vmax)
	plt.colorbar()
	ll = ["%s (%i)" % (labs[i], i+1) for i in range(len(labs))]
	plt.xticks(range(len(labs)), ["%i" % (i+1,) for i in range(len(labs))])
	plt.yticks(range(len(labs))[::-1], ll)

def show_icd(dm, labs, fig = 1):
	'''
	dm: DistMat(N), labs: N-[ of s, fig: i -> None (draws in matplotlib figure(fig) )
	
	Plots the distance matrix dm as an image in figure fig, labeling the rows 
	with the strings in labs. 
	
	'''
	f = plt.figure(fig)
	plt.clf()
	_draw_icd(dm, labs)
	f.canvas.draw()

def _gettreelinks(t):
	"""Internal, used by show_icd_ld"""''
	nn = len(t)+1
	ld = np.zeros((nn,nn))
	for i in range(1,nn):
		for j in range(i):
			d = clust.treedist_links(t, i, j)
			ld[i,j]=d
			ld[j,i]=d
	return ld

def show_icd_ld(dm, labs, fig=1):
	"""
	Like show_icd, but instead of showing the inter-class distance matrix it 
	shows the graph link distance. In this matrix, the i,j element is the 
	number of edges that must be followed to get from i to j in the DAG 
	generated by median-distance heirarchial clustering. It is thus in one 
	sense an ordinal version of the raw distance matrix. 
	"""
	f = plt.figure(fig)
	plt.clf()
	t = clust.dtree(dm)
	tldm = _gettreelinks(t)
	_draw_icd(tldm, labs)
	f.canvas.draw()
	
def _tsort(a, l):
	"""
	a: DistMat(N), l: N-[ of s -> DistMat(N), N-[ of s
	
	creates a new matrix and list of names sorted using tree clustering
	"""
	
	t = clust.dtree(a)
	co = clust.treesort(t)
	l2 = [l[i] for i in co]
	a2 = np.zeros_like(a)
	for i in range(a.shape[0]):
		for j in range(a.shape[1]):
			a2[i, j] = a[co[i],co[j]]
	return (a2, l2)

def rcell(cell):
	"""
	cell:i | CellExp -> (CellExp, s)
	
	convert cell input which could be a document or a cell id integer to a 
	document and name
	"""
	if type(cell) == int:
		d = load(cell)
		cn ="cell%i" % cell
	else:
		d = cell
		try:
			cn= "cell%i" % (d['docsource.cell'],)
		except TypeError:
			cn = d['docsource.cell']
	return (d, cn)
	
def show_cell_icd(cell, mode='vdps', q=6000, fig=1, 
                  save="~/Dropbox/wsu/roc/icd", suf='', tsort=1,
                  cn=None):
	"""
	cell: i, mode: DistMode, q: DistQ, fig: i, save: DirName, suf: s, tsort:i
		-> None (draws in matplotlib figure fig, may write a file)
	
	Shows the inter-class distances for each stimulus in each condition, and 
	also the clustering diagram. Draws a figure. If save is non-empty, then
	also writes a file in that directory, named:
	"c%i_icd_%s%i%s.png" % (cell, mode, int(q),suf)
	
	tsort is a flag. If true, sorts the dm using clust.treesort
	"""
	d, cn = rcell(cell)
	f = plt.figure(fig)
	plt.clf()
	plt.figtext(0,.98,"%s, mode=%s, q=%.2g" % 
	            (cn, mode, q), backgroundcolor='white')
	plt.subplot(121)
	labs = stimnames(d)
	a1 = dist_discrim_incond(d['cond1'],mode, q, len(labs))
	a2 = dist_discrim_incond(d['cond2'],mode, q, len(labs))
	vmin = min([a1.min(), a2.min() ])
	vmax = max([a1.max(), a2.max()])
	if tsort:
		olabs = labs
		a1, labs = _tsort(a1, labs)
	_draw_icd(a1, labs, vmin, vmax)
	plt.title("Inter-class distances, Condition 1")
	plt.subplot(122)
	plt.title("Inter-class distances, Condition 2")
	if tsort:
		if tsort > 1:
			a2, labs = _tsort(a2, olabs)
		else:
			inds = [olabs.index(l) for l in labs]
			na2 = np.zeros_like(a2)
			for i in range(a2.shape[0]):
				for j in range(a2.shape[1]):
					na2[i, j] = a2[inds[i],inds[j]]
			a2 = na2
	_draw_icd(a2, labs, vmin, vmax)
	f.canvas.draw()
	if save:
		save = os.path.expanduser(save)
		fn = os.path.join(save, "%s_icd_%s%i%s.png" % (cn, mode, int(q),
		                                                suf))
		plt.savefig(fn)

def show_tree(t, labs=None, fig=1):
	"""
	t: Tree(N) | s, labs: N-[ of s, fig: i -> None (draws in MPL figure fig)
	
	Plots an image of tree t made with graphviz. This is a display wrapper for 
	vis.dot2img and clust.tree2dot
	
	If t is a string to start with, it should be in the dot language (so this
	function can be used to plot other types of trees than those made by 
	tree2dot). In this case labs has no effect
	
	"""
	if not type(t) in [str, unicode]:
		t = clust.tree2dot(t, labs)
	f = plt.figure(fig)
	plt.clf()
	im = vis.dot2img(t)
	plt.imshow(im)	
	a = f.get_axes()[0]
	a.yaxis.set_visible(False)
	a.xaxis.set_visible(False)
	f.canvas.draw()

def show_ctree(ct, labs=None, fig=1):
	if labs == None:
		labs = max([max(c) for c in ct[0]])+1
	if type(labs) == int:
		labs = map(str, range(labs))
	s = clust.ctree2dot(ct[0], labs, ct[1])
	show_tree(s,fig=fig)

def show_cell_clust(cell, mode='vdps', q=6000, fig=1,
                    save="/home/gic/Dropbox/wsu/clust", suf='', cn=None):
	"""
	cell: i, mode: DistMode, q: DistQ, fig: i, save: DirName, suf: s
		-> None (draws in matplotlib figure fig, may write a file)
	
	Shows the clustering tree (as show_tree) for each stimuli in each condition,
	and also the clustering diagram. Draws a figure. If save is non-empty, then
	also writes a file in that directory, named: "c%i_clust_%s%i%s.png" % (cell,
	mode, int(q),suf)
	
	"""
	d, cn = rcell(cell)
	f = plt.figure(fig)
	plt.clf()
	plt.figtext(0,.98,"%s, mode=%s, q=%.2g" % (cn, mode, q), backgroundcolor='white')
	labs=stimnames(d)
	a1 = dist_discrim_incond(d['cond1'],mode, q, len(labs))
	a2 = dist_discrim_incond(d['cond2'],mode, q, len(labs))
	sb= plt.subplot(121)
	plt.title("Condition 1")
	im = vis.dot2img(clust.tree2dot(clust.dtree(a1), labs))
	plt.imshow(im)	
	sb.yaxis.set_visible(False)
	sb.xaxis.set_visible(False)
	sb = plt.subplot(122)
	plt.title("Condition 2")
	im = vis.dot2img(clust.tree2dot(clust.dtree(a2), labs))
	plt.imshow(im)	
	sb.yaxis.set_visible(False)
	sb.xaxis.set_visible(False)
	f.canvas.draw()
	if save:
		save = os.path.expanduser(save)
		fn = os.path.join(save, "%s_clust_%s%i%s.png" % (cn, mode, int(q),
		                                                  suf))
		plt.savefig(fn)

def dropsamp(cond, ndrop):
	"""
	cond: IntIO, ndrop: i -> IntIO
	
	Return a document like cond, but with ndrop samples removed at random
	
	"""
	n = len(cond['stims'])
	dr = np.random.permutation(n)
	nd = gd.Doc({'stims':[], 'evts':[]})
	for i in range(n-ndrop):
		nd['stims'].append(cond['stims'][dr[i]])
		nd['evts'].append(cond['evts'][dr[i]])
	return nd

def drop_tree(cond, mode='vdps', q=6000, nbs=20, 
                   ndrop=1, N=14):
	trees = []
	for _ in range(nbs):
		d = dropsamp(cond, ndrop)
		dm = dist_discrim_incond(d, mode, q, N)
		if np.any(np.isnan(dm)):
			print('Warning: subsample changes classes. Discarding')
			continue
		trees.append(clust.dtree(dm))	
	return trees

def bootstrap_tree(cond, mode='vdps', q=6000, nbs=20, 
                   ndrop=1, show = 0, names = None):
	"""
	cond: IntIO, mode:DistMode, q: DistQ, nbs:i, ndrop:i -> 
	
	"""
	if not names:
		sids = tuple(sorted(set(cond['stims'])))
		names = ['s%i' % i for i in sids]
	trees = drop_tree(cond, mode, q, nbs, ndrop, len(names))
	ft, nr = clust.mrtree(trees)
	if show:
		s = clust.ctree2dot(ft, names, nr)
		show_tree(s,fig=show)
	return (ft, nr)

def random_dm(n, minv=0, maxv=1):
	"""
	n:N of i, minv:M of x, maxv:X of x :: X>M ->  N,N-#
	
	return a uniform random distance matrix on the indicated range (symetric,
	diagonal values are 0, off diagonal values are uniform random numbers
	"""
	dm = np.random.uniform(minv, maxv, (n,n))
	for i in range(n):
		dm[i,i] = 0.0
		for j in range(i):
			dm[j, i] = dm[i,j]
	return dm 

def random_bstrap_tree(mode='vdps', q=6000, nbs=20, ndrop=1, nstims=14, 
                       npres=20, winlen=200000, nspikes=5):
	rt = nspikes/(float(winlen)/1e6)
	d = tests.cdoc((tests.AC(winlen, rt),), nstims, npres)['condhpois']
	return bootstrap_tree(d, mode, q, nbs, ndrop, 0, None)
	
def tree_stability(d, mode='vdps', q=6000, nbs = 20, ndrop=5):
	"""
	d: CellExp
	"""
	ts = []
	nd = d.copy()
	astim = d['cond1.stims'] + d['cond2.stims']
	aevts = d['cond1.evts'] + d['cond2.evts']
	names = stimnames(d)
	nd['condAll'] = {'stims':astim, 'evts':aevts}
	ts = []
	for i, c in enumerate(['cond1', 'cond2', 'condAll']):
		ft, nr = bootstrap_tree(nd[c], mode, q, nbs, ndrop, 0, names)
		ts.append((ft, nr))
	ss = [np.array(ts[i][1]).sum() for i in range(3)]
	sm = 2*ss[2] / (ss[0] + ss[1])
	return (sm, ts)

def droptree_groups(cd, mode='vdps', q=6000, nbs=10, ndrop=5):
	"""
	cd : CellDoc other args for drop_tree
	
	returns a document that contains a KeySet t of documents with "_" a tree2tup
	of a clustering tree, "cell" a cell name from cd, "cond" a condition name.
	These represent the result of running several sample drop clustering trials
	(as done by drop_tree), and labeling the resulting trees for cell and
	condition
	
	"""
	cn = cd.keys(0, 'cell')
	conds = ['cond1', 'cond2']
	d = gd.Doc()
	names = [cd[cn[0]]['stimclasses'][k]['file'] for k in 
	         cd[cn[0]]['stimclasses'].keys(sort=True)]
	i = 1
	for cell in cn:
		for cond in conds:
			trees = drop_tree(cd[cell][cond], mode, q, nbs, ndrop, len(names))
			for t in trees:
				d['t%i' % i] = {'_':clust.tree2tup(t), 'cell':cell, 
				                'cond':cond}
				i+=1
	d['names'] = names
	return d

def _pairwise(l, f):
	z = f(l[0], l[1])
	dm = {}
	for i in range(len(l)-1):
		for j in range(i, len(l)):
			dm[(i,j)] = f(l[i], l[j])
	return dm

def _dmfrompw(pw):
	z = 0
	for k in pw:
		z = max(z, max(k))
	z+=1
	dm = np.zeros((z,z))
	for k in pw:
		dm[k[0], k[1]] = pw[k]
		dm[k[1], k[0]] = pw[k]
	return dm

def _grpstats(dtg, pw):
	# first index is cell is the same  (0, 1)
	# second is condition is the same (0, 1)
	zz = [ [[], []],
	       [[], []] ]
	for k in pw:
		t1 = "t%i" % (k[0]+1,)
		t2 = "t%i" % (k[1]+1,)
		i = int(dtg[t1]['cell'] == dtg[t2]['cell'])
		j = int(dtg[t1]['cond'] == dtg[t2]['cond'])
		zz[i][j].append(pw[k])
	return zz
		

def intergroup_zss(dtg):
	"""dtg is output from droptree_groups"""
	ks = dtg.keys(0, 't', sort=True)
	ts = [clust.tup2tree(dtg[k]['_']) for k in ks]
	pw = _pairwise(ts, lambda x,y:clust.zssdist(x, y, dtg['names']))
	zz = _grpstats(dtg, pw)
	return (zz, pw)

def show_intergroup_zss(zz, pw, dtg, nbs=10, nbars = 10):
	f = plt.figure(1)
	plt.clf()
	plt.hist(zz[0][0], nbars, color = 'r', normed=True,
	         label = 'different cell and condition')
	plt.hist(zz[1][1], nbars, color = 'b', normed=True,
	         label = 'same cell and condition')
	plt.hist(zz[0][1], nbars, color = (1, .5, 0), normed=True,
	         alpha=.6, label = 'different cell')
	plt.hist(zz[1][0], nbars, color = 'g', normed=True,
	         alpha=.6, label = 'different condition')
	plt.legend(loc='best')
	f.canvas.draw()
	f = plt.figure(2)
	plt.clf()
	dm = _dmfrompw(pw)
	plt.imshow(dm)
	plt.colorbar()
	x = np.arange(0, dm.shape[1], nbs*2)+nbs
	cn = [dtg['t%i.cell' % i] for i in x] 
	plt.xticks(x, cn, rotation = 'vertical')
	f.canvas.draw()
	
def full_intergroup_zss(cd, mode='vdps', q=6000, nbs=10, ndrop=5):
	dtg = droptree_groups(cd, mode, q, nbs, ndrop)
	zz, dm = intergroup_zss(dtg)
	show_intergroup_zss(zz, dm, dtg)
	return (zz, dm, dtg)

## This method of tree comparison is depricated in favor of ZSS 	
##def all_tree_stability(cd, mode='vdps', q=6000, nbs = 20, ndrop=5, show=1):
##	dname = os.path.expanduser("~/Dropbox/wsu/clust/stability")
##	suf = ''
##	od = gd.Doc()
##	for cn in cd:
##		if not cn.startswith('cell'):
##			continue
##		sm, ts = tree_stability(cd[cn], mode, q, nbs, ndrop)
##		od[cn] = {'stability':sm, 'ctrees':ts}
##		print('%s : %.3g' % (cn, sm))
##		if show:
##			names = stimnames(cd[cd.find('_', kp=':^cell', depth=1).next()])
##			for i, c in enumerate(['cond1', 'cond2', 'condAll']):
##				ft, nr = ts[i]
##				f=plt.figure(1)
##				plt.clf()
##				plt.title("Bootstrap Majority Tree, %s" % c)
##				s = clust.ctree2dot(ft, names, nr)
##				im = vis.dot2img(s)
##				plt.imshow(im)	
##				sb = f.get_axes()[0]
##				sb.yaxis.set_visible(False)
##				sb.xaxis.set_visible(False)
##				f.canvas.draw()
##				fn = os.path.join(dname, "%s_%s_mrtree%s%i%s.png" % (cn, c, mode,
##					                int(q), suf))
##				plt.savefig(fn)
##	return od
##
##def trees_across_cells(cd, mode='vdps', q=6000):
##	trees = []
##	names = None
##	for cn in cd:
##		if not cn.startswith('cell'):
##			continue
##		if names is None:
##			names = stimnames(cd[cn])
##		for cond in ['cond1', 'cond2']:
##			dm = dist_discrim_incond(cd[cn][cond], mode, q, len(names))	
##			trees.append(clust.dtree(dm))
##	ft, nr = clust.mrtree(trees)
##	s = clust.ctree2dot(ft, names, nr)
##	show_tree(s)
##	
##def tree_sim_per_q(cell, mode='vdps', qr = [1, 20000, 4000]):
##	pass
##
##def do_all_icd(cd, q = 6000, mode='vdps', save="/home/gic/Dropbox/wsu/clust", 
##               suf=''):
##	"""
##	cd: CellDoc
##	(q, mode, save, suf): as for show_cell_icd, cells: [ of i -> None 
##		(writes files and draws in MPL figure 1)
##	
##	Does show_cell_icd and show_cell_clust with the indicated (q, mode, save, 
##	suf) and fig=1, for every cell in cd. The point of this is to save 
##	all the image files. Drawing in figure 1 is rather useless, since it 
##	overwrites the things it draws often. Probably should be modified to
##	use a file based MPL backend instead, eventually.
##	
##	"""
##	for cell in cd:
##		show_cell_icd(cd[cell], mode, q, cn = cell)
##		show_cell_clust(cd[cell], mode, q, cn = cell)
##
##def icd_treediff(cell, mode, q):
##	"""
##	cell: i | CellExp, mode: DistMode, q: DistQ -> float
##	
##	"""
##	if type(cell) == int:
##		d = load(cell)
##	else:
##		d = cell
##	labs = stimnames(d)
##	a1= dist_discrim_incond(d['cond1'],mode, q, len(labs))
##	a2= dist_discrim_incond(d['cond2'],mode, q, len(labs))
##	t1 = clust.dtree(a1)
##	t2 = clust.dtree(a2)
##	td = clust.cmptrees(t1, t2)
##	return (td.sum() - np.diag(td).sum())/float(td.shape[0])
##
##def scan_cell_tdiff(cell, modes=['ed_bin', 'vd', 'vdps', 'ed_ist'], 
##                    qs = [1000, 2000, 5000, 10000, 20000, 100000],
##                    plot=True, save=''):
##	"""
##	cell: i, modes: [ of DistMode, qs: [ of DistQ, plot: t, save: FileName ->
##		N-M-# of x::N==len(modes), M==len(qs)    (may plot in MPL fig 1 
##		and may save a file)
##	
##	returns an array of the values of icd_treediff(cell, mode, q) for each 
##	mode in modes and each q in qs.
##	
##	if plot is true, draw a graph of the results, with one line per mode, and
##	one point per q. If save is non-empty, also save the plot
##	"""
##	d = load(cell)
##	out = np.zeros((len(modes), len(qs)))
##	if plot:
##		f = plt.figure(1)
##		plt.clf()
##		qs = np.array(qs)
##	for i in range(out.shape[0]):
##		for j in range(out.shape[1]):
##			out[i,j] = icd_treediff(cell, modes[i], qs[j])
##		if plot:
##			plt.plot(qs, out[i,:], label = modes[i])
##	if plot:
##		plt.legend(loc='best')
##		f.canvas.draw()
##		if save:
##			plt.savefig(save)
##	return out
##
##def scan_many_cell_tdiffs(cells=GOOD, 
##                          modes=['ed_bin', 'vd', 'vdps', 'ed_ist'], 
##                          qs = [1000, 2000, 5000, 10000, 20000, 100000],
##                          save='~/Dropbox/wsu/roc/tdiffs'):
##	"""
##	cells: [ of i, (modes, qs) as for scan_cell_tdiff, save: DirName -> 
##		None (writes files in save, draws in MPL figure 1)
##
##	Runs scan_cell_tdiff for each cell in cells, with a save argument of 
##	save/cell#.png
##	
##	"""
##	save = os.path.expanduser(save)
##	for cell in cells:
##		cn = "cell%i.png" % cell
##		scan_cell_tdiff(cell, modes, qs, True, os.path.join(save, cn))
##	
def pairwise_ROC(d, cond='cond1',mode='ed_bin', q=20000, nroc=100):
	"""
	d: CellExp, cond: CondName(d), mode: DistMode, q: DistQ, nroc: i ->
		ret: {(i,i) of Roc(nroc)
	
	Constructs every pairwise comparison between stimuli in d[cond] as a 
	classify.roc ROC curve. Returns a dictionary of pairs (i, j) onto Rocs, 
	where (i, j) are the two stimuli being separated. 
	"""
	dm = dist.dist(d[cond]['evts'], mode, q)
	st = np.array(d[cond]['stims'])
	stn = _stimnames(d, cond, True)
	stpairs = clust.pairs(sorted(stn))
	ret = {}
	for p in stpairs:
		s1 = np.nonzero(st == stn[p[0]])[0]
		s2 = np.nonzero(st == stn[p[1]])[0]
		ind = np.union1d(s1, s2)
		scla = np.array([(x in s1 or 2) for x in ind])
		dm_p = dm[ind,:][:, ind]
		ret[p] = roc(scla, dm_p, nts = nroc)
	return ret
	
def show_pairROC(pr, fig=1):
	"""
	pr: ret from pairwise_ROC, fig: i -> None (draws in MPL figure fig)
	
	Draws every curve in the pairwise_ROC scan pr. Usually rather useless. 
	"""
	f = plt.figure(fig)
	plt.clf()
	for k in pr:
		plt.plot(pr[k][:,1], pr[k][:,0], label = k[0]+'/'+k[1])
	plt.legend(loc='best')
	f.canvas.draw()
	
def _areaunder(a, npts=20):
	"""
	a: Roc, npts: i -> x
	
	Estimates the area under the ROC curve a by resampling this curve 
	uniformly onto npts samples and taking the sum (note that this doesn't 
	account for the bin width, so you need to devide the result by npts to 
	get a real area. It can be used to order ROCs by "goodness", though, 
	provided all measurements are made with the same value of npts
	
	"""
	aa = np.row_stack( [np.array([[0,0]]), a, np.array([[1,1]]) ])
	x = np.unique(a[:,1])
	y = np.zeros_like(x)
	for i in range(x.shape[0]):
		y[i] = aa[np.nonzero(aa[:,1]==x[i]),0].max()
	aa = np.interp(np.linspace(0,1,npts), x, y)
	return aa.sum()

def show_ordered_pairROC(pr, r = (0, 10), fig=1):
	"""
	pr: ret from pairwise_ROC, r:(i, i), fig: i -> [ of (i,i) (draws a figure)
	
	Orders all the ROCs in pr according to the area under the curve. Plots the
	range range(r[0], r[1]) of curves from this ordering in MPL figure fig. 
	Returns the list of pairs that correspond to the curves plotted. 
	
	"""
	def _cp(k1, k2):
		return cmp(_areaunder(pr[k2]), _areaunder(pr[k1]))
	ks = sorted(pr, cmp=_cp)
	ks = [ks[n] for n in apply(range, r)]
	#vs = [_areaunder(pr[k]) for k in ks]
	f = plt.figure(fig)
	plt.clf()
	for k in ks:
		p = plt.plot(pr[k][:,1],pr[k][:,0], label = k[0]+'/'+k[1], linewidth=3)
		c = p[0].get_color()
		#plt.fill_between( pr[k][:,1],0, pr[k][:,0], color=c)
	plt.legend(loc='best')
	f.canvas.draw()
	return ks

