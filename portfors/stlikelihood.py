#!/usr/bin/env python -3
# encoding: utf-8

#Created by Graham Cummins on 

# Copyright (C) 2011 Graham I Cummins
# This program is free software; you can redistribute it and/or modify it
#underthe terms of the GNU General Public License as published by the Free
#Software Foundation; either version 2 of the License, or (at your option) 
#any later version.

#This program is distributed in the hope that it will be useful, but WITHOUT 
#ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
#FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

#You should have received a copy of the GNU General Public License along with
#this program; if not, write to the Free Software Foundation, Inc., 59 Temple
#Place, Suite 330, Boston, MA 02111-1307 USA


from __future__ import print_function, unicode_literals
from mixmod import mmcall, evaluate
from gicdat.enc import flat
from gicdat.util import traverse
import numpy as np
import matplotlib.pyplot as plt
from selectUI import stimnames
from gicdat.doc import Doc

"""
cd -> CellDoc
ce -> CellExp
cond -> IntIO


"""

def likelihood(gmm, st, mode = 'log'):
	"""
	returns a measure of the probability of spike train st under model gmm.
	mode may be:
	
	log: conventional log likelihood. This measure makes the fewest assumptions,
		but is also monotonically decreasing with len(st) (each actual event is
		improbable, so any possible long event sequence is massively improbable)
	av: log of the average of the activation for each spike. Essentially assumes
		all spikes are independent. A spike train with many spikes, a few of 
		which are in strongly excluded regions of the model, may still have
		relatively good likelihood. 
	la: the log measure, devided by the length of the spike train.
	
	"""
	acts = evaluate(gmm, st)
	if mode == 'log':
		return np.log(acts).sum()
	elif mode == 'av':
		return np.log(acts.mean())
	elif mode == 'la':
		return np.log(acts).sum()/len(st)
	

def spikes_from(cond, stim, flatten = True):
	"""
	return a flat list of every spike in cond['evts'] resulting from stmulus 
	stim (int). 
	
	"""
	resps = [cond['evts'][i] for i in range(len(cond['evts'])) 
	         if cond['stims'][i] == stim]
	if flatten:
		return flat(resps)
	else:
		return resps

def response_densities(cond, mcent = 8, nrep=3, stims = None, 
                       sppt = 163000, dsd=10000):
	"""
	return a list l such that l[i] = is a gmm of the responses to stimulus i.
	
	Since these are 1D models, the underlying mixmod call always uses PkLkCk,
	but mcent specifies the max number of centers to try, nrep the number of 
	repeats to use.
	
	stims may be a list, which restricts the set of stimuli that are modeled
	(by default, it is all presented stimuli).
	
	If the number of spikes evoked by a stimulus is very small, mixmod errors
	will result. This function will not try to calculate a model based on fewer
	than 2 spikes per center, reguardless of the value of mcent. For response
	groups with at least two spikes, mcent may be reduced. Responses with no
	spikes are modeled with a 0-component model (the dictionary
	{'components':(), 'support':sppt}. mixmod.evaluate on such a model will
	return a uniform probability if 1/sppt, so this behavior is equivalent to a
	uniform prior over a region of size sppt. Responses with exactly 1 spike are
	modeled as a single center, with mean at the time of that spike, and
	standard deviation specified by the free parameter dsd.
	
	This function adds the key "responses" to the resulting model dictionaries,
	containing the source spike trains
	
	"""
	if stims is None:
		stims = set(cond['stims'])
	l = [None for _ in range(max(stims)+1)]
	for s in stims:
		sts = spikes_from(cond, s, False)
		sf = flat(sts)
		if len(sf)>1:
			mc = int(min(mcent, np.ceil(len(sf)/2.0)))
			l[s] = mmcall(np.array(sf)[:,np.newaxis], range(1, mc+1),
			              reps = nrep)	
		elif sf:
			l[s] = {'proportions': (1.0,), 'c0': {'cov': ((dsd**2,),), 
			                                    'mean': (sf[0],)}, 
			      'bic': 0, 'components': ('c0',), 'partition': (0,)}
		else:
			l[s] = {'components':(), 'support':sppt}
		l[s]['responses'] = sts
			
	return l

def cell_resp_d(ce, **kw):
	"""return a Doc with keys for conditions in ce, each containing a doc
	representing response_densities for that condition (these docs use the 
	name of each stimulus to key the model of that stimulus as returned by
	responses_densities). Keywords are passed to responses_densities"""
	conds = ce.keys(0, 'cond')
	r = Doc()
	names = stimnames(ce)
	for c in ce.keys(0, 'cond'):
		m = response_densities(ce[c], **kw)
		md = Doc()
		for i, sn in enumerate(names):
			md[sn] = m[i]
		r[c] = md
	return r

def group_resp_d(cd, **kw):
	"""returns a Doc with cell keys as cd, each containing a cell_resp_d for
	that cell"""
	nd = traverse(cd, 'cell', cell_resp_d, (), kw, 'patch')
	return nd

def show_cell_density(rd, xran = (0, 143000), 
                      overlay =False):
	""" 
	rd is the result of cell_resp_d.  xran is the x range of the plot
	Will glitch if the various conditions in rd used different stimulus sets - 
	avoid this. If overlay, plot all data in one subplot in different colors
	
	"""
	f  = plt.figure(1)
	plt.clf()
	cols = 'brgkcy'	
	x = np.arange(xran[0], xran[1])
	pad = .9
	conds = rd.keys(0, 'cond')
	stims = rd[conds[0]].keys(sort=cmp)
	for i, sn in enumerate(stims):
		for j, cn in enumerate(conds):	
			if overlay:
				col = cols[j]
				siz = 10-2*j
			else:
				plt.subplot(1, len(conds), j+1)
				col = 'k'
				siz = 6
			sf = rd[cn][sn]['responses']
			mm = evaluate(rd[cn][sn], x)
			if mm.max()-mm.min()>0:
				mm = pad*mm/mm.max()
			plt.plot(x, mm+i, linewidth=3, color=col)
			sp = pad/(len(sf)+1)
			for k in range(len(sf)):
				if sf[k]:
					y = np.ones(len(sf[k]))*(k+1)*sp + i
					plt.plot(sf[k], y, '.', color = col, markersize=siz)
	if not overlay:				
		plt.subplot(1, len(conds),1)
	plt.xlim(xran)
	plt.yticks(range(len(stims)), stims)
	if not overlay:
		f.subplots_adjust(left=.04, right=.99, bottom=.03, top = .94, wspace=.05)	
		for i in range(2, len(conds)+1):	
			sp = plt.subplot(1, len(conds),i)
			plt.xlim(xran)
			sp.yaxis.set_visible(False)
			sp.xaxis.set_visible(False)
	f.canvas.draw()	

def preference(rd, st, conds, stims, mode='log'):
	"""
	rd: response density doc, st: spike train, conds: list of keys, 
	stims: list of keys, mode: parameter for likelihood()
	
	return an array resp of shape (len(conds), len(stims)) containing at i,j the
	result of likelihood(rd[conds[i]][stims[j]], st, mode)
	
	"""
	resp = np.zeros((len(conds), len(stims)))
	for i in range(resp.shape[0]):
		for j in range(resp.shape[1]):
			resp[i,j] = likelihood(rd[conds[i]][stims[j]], st, mode)
	return resp


def pref_hists(grd, mode='log'):
	s= []
	c2 = []
	o = []
	conds = ['cond1', 'cond2']
	for cell in grd.keys(0, 'cell'):
		rd = grd[cell]
		stims = rd['cond1'].keys(sort=cmp)
		for i, sn in enumerate(stims):
			for st in rd['cond1'][sn]['responses']:
				if st:
					p = preference(rd, st, conds, stims, mode)
					s.append(p[0,i])
					c2.append(p[1, i])
					for ri in range(p.shape[1]):
						if ri!=i:
							o.extend(p[:,ri])
	return (s, c2, o)

def classify(grd, mode='log'):
	cerr = []
	conds = ['cond1', 'cond2']
	for cell in grd.keys(0, 'cell'):
		rd = grd[cell]
		stims = rd['cond1'].keys(sort=cmp)
		for i, sn in enumerate(stims):
			for st in rd['cond1'][sn]['responses']:
				if st:
					print(i, st)
					p = preference(rd, st, conds, stims, mode)
					print(p)
					c1i = np.argmax(p[0,:])
					c1e = p[0,c1i]-p[0, i]
					c2i = np.argmax(p[1, :])
					c2e = p[1,c2i]-p[1, i]
					print(c1i, c2i)
					cerr.append( (c1e, c2e) )
	return np.array(cerr)
	
	