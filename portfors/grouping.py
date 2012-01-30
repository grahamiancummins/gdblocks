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

from __future__ import print_function, unicode_literals
import numpy as np
import dist, clust
from classify import jitter

def idents(l):
	'''
	l is a sequence of hashables. Return is a sequence of the same length, of integers. 
	Integers represent the different identies of the elements in l (e.g. if the first 
	and 5th element of l are structurally equal, then the return will have elements 0 and 4
	equal to the same value (which in this example will also be 0, since ids happen to be the
	index of the first example of the pattern. 
	
	'''
	found={}
	ids = []
	for id, v in enumerate(l):
		if not v in found:
			found[v] = id
		ids.append(found[v])
	return ids
	
def identwith(l, f):
	'''
	like idents, but instead of using structural equality to determine if two things are the same, use the function 
	f, which should take two tuples and return boolean (representing if these tuples should be classed as the same)
	'''
	found = {}
	ids = []
	for id, v in enumerate(l):
		for k in found:
			if f(k, v):
				ids.append(found[k])
				break
		else:
			found[v] = id
			ids.append(id)
	return ids

def identdthresh(dm, t, rel = True, mode='first'):
	'''
	groups elements of dm that have distance less than t
	
	If rel is True, t is taken to be a fraction of range (the 
	actual threshold is min + t*(min -max), for min and max of the
	dm)
	
	Mode specifies how to calculate distances. It is a key into 
	clust.DTHRESH
	
	
	'''
	#FIXME: slow
	if rel:
		mid = dm.min()
		mad = dm.max()
		t = mid + (mad - mid)*t
	ids =  clust.DTHRESH[mode](dm, t)
	return ids

def identdclust(doc, dm, n):
	members = [[i] for i in range(dm.shape[0])]
	_, members = clust.hca(dm, members, n)
	ids = np.zeros(dm.shape[0], np.int32)
	for i, m in enumerate(members):
		for rid in m:
			ids[np.array(rid).astype(np.int32)]=i
	return doc.fuse({'evts':ids})
	
def bins(t, q, counts=True):
	r = [int(round(v/float(q))) for v in t]
	rvs = tuple(sorted(set(r)))
	if counts:
		return tuple([(v, r.count(v)) for v in rvs])
	else:
		return rvs

def binrep(cond, q=5000, counts=True):
	out = [bins(et, q, counts) for et in cond['evts']]
	return cond.fuse({'evts':idents(out)})	

def _cematch(s1, s2, q):
	if len(s1)==len(s2) and (len(s1)==0 
							or np.abs(np.array(s1) -np.array(s2)).max() < q):
		return True
	else:
		return False

def closeenough(cond, q):
	z = cond.copy()
	z.set('evts',  identwith(cond['evts'], lambda x, y:_cematch(x, y, q)), False)
	#calling fuse wastes core time on type checking 
	return z # cond.fuse({'evts':identwith(cond['evts'], lambda x, y:_cematch(x, y, q))})
	
def distthresh(cond, q, dmethod='bin', thresh=.9, nulls = None, trel=False):
	lot = cond['evts']
	dists = dist.dist(lot, q, dmethod, nulls)
	ids = identdthresh(dists, thresh, trel)
	return cond.fuse({'evts':ids})

def clustdm(cmeth, dm, nclust, cargs={}):
	if cmeth == 'tree':
		t = clust.dtree(dm, **cargs)
		clsts = t.cut(nclust)
	elif cmeth == 'med':
		clsts = clust.mediods(dm, nclust, **cargs)
	else:
		raise StandardError("don't know clustering method %s" % cmeth)
	return clsts

def dclust(cond, q=1, dmethod='ibl', nulls=None,
		cmeth='tree', nclust=16, cargs={}):
	dists = dist.dist(cond['evts'], dmethod, q, nulls)
	clsts = clustdm(cmeth, dists, nclust, cargs)
	return cond.fuse(cond.new(evts=clsts))

def vclust(cond, q=1, vrep='irate', vargs={}, cmeth='kmeans', nclust=16, cargs={}):
	evts = dist.VREPS[vrep](cond['evts'], **vargs)
	if cmeth == 'tree':
		t = clust.vtree(evts, **cargs)
		clsts = t.cut(nclust)
	elif cmeth == 'kmeans':
		clsts = clust.kmeans(evts, nclust, **cargs)
	elif cmeth == 'mixmod':	
		clsts = clust.mixmodpartition( evts, nclust, **cargs)
		#cargs should contain eg. {'model':"Gaussian_pk_Lk_Bk", reps:1} 
	else:
		raise StandardError("don't know clustering method %s" % cmeth)
	return cond.fuse(cond.new(evts=clsts))
	
REPRS = {
	'bins':binrep,
	'ce':closeenough,
	'dthresh':distthresh,
	'dclust':dclust,
	'vclust':vclust,
	'none':lambda c, q: c
	
}

def classifyresp(lot, meth, q, **kwargs):
	return REPRS[meth](lot, q, **kwargs)
