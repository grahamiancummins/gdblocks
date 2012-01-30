#!/usr/bin/env python -3
# encoding: utf-8

#Created by Graham Cummins on May 10, 2011

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
'''
all input documents are of the sort generated by timingInfo.getcell
They contain a set of keys "cond1, cond2 ...." 
Each of these contain "evts" (list of tuples), and "stims" (list of ints)
'''

import numpy as np
from gicdat.enc import trange
import dist
import matplotlib.pyplot as plt

def resps(d, cond, stim):
	'''
	return a list of all the event sequences is cond (condition document) that
	correspond to presentations of stimulus stim (integer).

	'''
	return [d[cond+'.evts'][i] for i in range(len(d[cond+'.evts'])) 
	        if d[cond+'.stims'][i] == stim]

def resprate(d, conds, stim):
	'''
	return the average number of spikes (float) evoked per presentation of stim
	(integer) in d (experiment document) cosidering only the coditions listed in
	conds (list of string keys)

	'''
	rs = []
	for c in conds:
		rs.extend(resps(d, c, stim))
	npres = len(rs)
	nspks = np.array([len(x) for x in rs]).sum()
	return float(nspks)/npres

def rank(d, conds=('cond1',)):
	'''
	return a sorted 2D array containing, in the first column, the stimulus
	numbers, and in the second, the values of resprate(d, conds, stim), for each
	stimulus found in d.

	'''
	stims= np.unique(np.concatenate([d[k+'.stims'] for k in conds]))
	rr = np.array([resprate(d, conds, s) for s in stims])
	inds = np.argsort(rr)[::-1]
	return np.column_stack([stims[inds], rr[inds]])

def rankdiff(d, pc = 'cond1', rc = 'cond2'):
	'''
	As "rank", but the second column contains the difference 
	resprate(d, (pc,), s) - resprate(d, (rc,), s) for each s that occurs in 
	both conditions pc and rc

	'''
	stims= np.intersect1d(d[pc + '.stims'], d[rc+'.stims'], False)
	r1 = np.array([resprate(d, [pc], s) for s in stims])
	r2 = np.array([resprate(d, [rc], s) for s in stims])
	rr = r1 -r2
	inds = np.argsort(rr)[::-1]
	return np.column_stack([stims[inds], rr[inds]])

def prefstim(d, good=1, conds = ('cond1', 'cond2'), diff=False):
	'''
	return two classes of stimuli, (good and bad) using rank (if diff is False)
	or rankdiff (if diff is true) as the criteria. If good is float, it is a
	fraction of stimuli to put in the good class. If it is int, it is the number
	of stimuli.

	'''
	if diff:
		r = rankdiff(d, conds[0], conds[1])
	else:
		r = rank(d, conds)
	if type(good) != int:
		thresh = good*r[0,1]
		good = 0
		while r[good, 1]>thresh:
			good+=1
	gc = r[:good, 0]
	oc = r[good:, 0]
	return (gc, oc)

def jitter(cond, sdev, insert=0, n = 30):
	'''
	add noise to all the spike trains in cond['evts'] (cond is a condition
	document). Each spike train is duplicated n times, and each duplicate is
	modified by adding random values from a normal distribution with mean 0 and
	std dev sdev. If insert is > 0, then it acts as a probability that a spike is
	added to the train as well (up to 3 spikes can be added, with probability
	insert, insert**2 insert**3)

	'''
	newc = cond.new()
	srange = trange(cond['evts'])
	st = []
	ev = []
	for i,t in enumerate(cond['evts']):
		stim = cond['stims'][i]
		l = len(t)
		for j in range(n):
			nt = tuple([int(x) for x in t])
			if l and sdev:
				jit = (np.random.randn(l)*sdev).astype(np.int32)
				nt =  tuple(np.array(nt) + jit)
			if insert:
				ads = np.random.random()
				nns = 0
				if ads < insert:
					nns+=1
					if ads < insert**2:
						nns+=1
						if ads < insert**3:
							nns+=1
				if nns:
					new = np.random.randint(srange[0], srange[1], nns)
					nt = tuple(sorted(nt + tuple(new)))
			ev.append(nt)
			st.append( stim )
	newc['evts'] = tuple(ev)
	newc['stims'] = tuple(st)
	return newc

def jitdoc(d, sdev, insert=0, n=30):
	nd = d.new()
	for k in d:
		if k.startswith('cond'):
			nd[k] = jitter(d[k], sdev, insert, n)
		else:
			nd[k] = d[k]
	return nd

def sclasses(stims, prefs):
	'''
	return a list that contains a 1 in position i if stims[i] is in prefs[0],
	and a 2 otherwise. Stims is a list of ints, prefs a tuple of 2 lists of
	ints)

	'''
	z = np.zeros(len(stims))
	for i in range(len(stims)):
		if stims[i] in prefs[0]:
			z[i] = 1
		elif stims[i] in prefs[1]:
			z[i] = 2
	return z

def classify(resp, scla, dm):
	'''
	Return a classification statistic for spike train "resp" into the classes
	specified by scla (as returned by sclasses), using the distance measure in
	matrix dm

	'''
	gc = np.array([i for i in range(len(scla)) if scla[i] == 1 and i!=resp])
	oc = np.array([i for i in range(len(scla)) if scla[i] == 2 and i!=resp])
	gcd = dm[gc, resp]
	ocd = dm[oc, resp]
	return ocd.mean() - gcd.mean()

def roc(scla, dm, nts = 100):
	'''
	construct an ROC curve for classification vector scla (as returned by
	sclasses) and distance matrix dm, with nts points sampled.
	'''
	cl = np.array([classify(i, scla, dm) for i in range(dm.shape[0])])
	mi = cl.min()
	ma = cl.max()
	thr = np.linspace(mi, ma, nts)
	pts = np.zeros((nts, 2))

	npos = float((scla == 1).sum())
	nneg = float((scla == 2).sum())
	for i, t in enumerate(thr):
		f = cl >=t
		pts[i,0]= np.logical_and(f == 1, scla == 1).sum()/npos
		pts[i, 1]= np.logical_and(f == 1, scla == 2).sum()/nneg
	return pts  

def sc_dm(d, cond='cond1',dmode='ed_bin', q=20000, good=1, scond=('cond1',), 
          doroc=100, nulls=None):
	'''
	construct a pair (scla, dm) for d[cond] using the arguments good, scond for
	prefstim, and q, dmode, nulls for dist.dist.

	If doroc, use these to build a doroc-point ROC curve and return that.
	Otherwise return the tuple (scla, dm).

	'''
	if scond == 'diff':
		pref = prefstim(d, good, ('cond1', 'cond2'), True)
	else:
		pref = prefstim(d, good, scond, False)
	scla = sclasses(d[cond]['stims'], pref)
	dm = dist.dist(d[cond+'.evts'], dmode, q, nulls)
	if doroc:
		return roc(scla, dm, doroc)
	else:
		return (scla, dm)

def rocvq(d, dmode='bin', qs=np.arange(1, 100000, 5000), good=1, 
          scond=('cond1',), nulls=None):
	'''
	scan over q in qs to find the best discrimination 
	'''
	nd = d.new()
	if scond == 'diff':
		pref = prefstim(d, good, ('cond1', 'cond2'), True)
	elif scond == 'man':
		stims= np.intersect1d(d['cond1.stims'], d['cond2.stims'], False)
		if type(good) == int:
			good = (good,)
		pref = (good, np.setdiff1d(stims, good)) 
	else:
		pref = prefstim(d, good, scond, False)
	nd['qs'] = tuple(qs)
	for cond in d.keys():
		if not cond.startswith('cond'):
			continue
		scla = sclasses(d[cond+'.stims'], pref)
		bvs = []
		for i, q in enumerate(qs): 
			dm = dist.dist(d[cond+'.evts'], dmode, q, nulls)
			r = roc(scla, dm, 100)
			nd[cond+'.roc%i' % i] = r
			bvs.append((r[:,0] - r[:,1]).max())
		nd[cond+'.bv'] = tuple(bvs)
	return nd

def bestdiscr(d, dmode='bin', qs = np.linspace(1,35000, 15), nulls=None):
	'''
	scan over stimuli to find the one that can be best discriminated
	'''
	stims= np.intersect1d(d['cond1.stims'], d['cond2.stims'], False)
	nd = d.new()
	nd['qs'] = tuple(qs)
	bsc1 = None
	bsc2 = None
	bvc1 = -np.inf
	bvc2 = -np.inf
	for s in stims:
		dd = rocvq(d, dmode, qs, good = (s,), scond='man', nulls=nulls)
		sn = 's%i' % s
		nd[sn] = dd
		vc1 = max(dd['cond1.bv'])
		vc2 = max(dd['cond2.bv'])
		if vc1>bvc1:
			bvc1 = vc1
			bsc1 = s
		if vc2 > bvc2:
			bvc2 = vc2
			bsc2 = s
	nd['c1best'] = bsc1
	nd['c2best'] = bsc2		
	return nd

		

	