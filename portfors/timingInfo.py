#!/usr/bin/env python
# encoding: utf-8

#Created by gic on Tue Oct 19 10:08:26 CDT 2010

# Copyright (C) 2010 Graham I Cummins
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later 
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA 02111-1307 USA
#

from mi import minf, minf_db
from gicdat.base import Transform
import dist
from gicdat.enc import flat
from grouping import classifyresp, clustdm, identdthresh
import clust
import numpy as np
from classify import jitter
import gicdat.doc as gd

def expandwindows(roi, start, stop, units=1.0):
	wins = []
	for w in roi:
		st, sp = w
		if type(st) in [str, unicode]:
			st = stop + eval(st)
		else:
			st = start + st
		if type(sp) in [str, unicode]:
			sp = stop + eval(sp)
		else:
			sp = start + sp
		wins.append((st*units, sp*units)) 
	return wins

def expandallwin(d, windows, units):
	ww = {}
	for s in d['stimclasses'].keys(0, 'stim'):
		sid = int(s[4:])
		sdict = d['stimclasses'][s]
		sst = sdict.get('onset') or 0.0
		sd = sdict.get('duration') or 0.0
		sd = sst+sd
		ww[sid] = expandwindows(windows, sst, sd, units)
	return ww

def _applywin(e, w):
	""" Internal. Used by window"""
	nes = []
	cos = 0
	for win in w:
		start, stop = win
		nes.extend([x-start+cos for x in e if x>=start and x<stop])
		cos = cos+stop-start
	return nes

def window(d, windows=[(0, 10), ('-10', '0')]):
	'''
	d: CellExp, windows: [ of (i|s, i|s) -> nd: CellExp
	
	For every condition in d, transform the events list to reflect a sequence of
	regions of interest. The function returns a new document containing these
	events. The input document is not modified. The regions are specified by
	"windows" as decscribed below. The mresulting events sequences are
	constructed by discarding any events that occur outside all windows, and
	adjusting the times of events so that the windows are treated as an adjacent
	tiling. This means that an event occuring 2ms into the first window will
	have time 2, and event 5 ms into the second window will have time 5+length
	of the first window, etc. Times passed to the "window" argument are in
	miliseconds. Times of the output spikes are still in samples (microseconds).
	This function should be applied before using resampling or binning (for
	example from timingInfo.PreProc), since the function assumes event times in
	microseconds.
	
	The "windows" argument is a list of tuples, (start, length), specifying 
	the start time and end time of each window, in miliseconds, reletive to event
	time 0 (which is typically the start of the stimulus, if data were loaded 
	with functions from this module or "explore"). 
	
	Either value may be written as a string. The string should evaluate to a
	number, but the fact that it is written at the string instructs the function
	to interpret it relative to the end of the stimulus (which is one of several
	times for different event series, depending on which stimulus they associate
	to). Thus, -10 refers to 10 ms before the start of the stimuls, but "-10"
	refers to 10 ms before the end.
	
	This function allows windows to be expressed in any order, to overlap, or to 
	duplicate. This means it can resample the same source spike. Note that the 
	algorithm for spike time reasignment will always mean that spikes from a 
	later window occur at later times in the returned response, even if they 
	are drawn from earlier times in the original. 
	
	'''
	nd = gd.Doc()
	ww = expandallwin(d, windows, 1000.0)
	for k in d.keys(0, 'cond'):
		nd[k]={}
		nevts = []
		for i, e in enumerate(d[k]['evts']):
			nevts.append(_applywin(e, ww[d[k]['stims'][i]]))
		nd[k]['evts'] = nevts
	return nd

def stimsets(io):
	ss = {}
	astims = np.array(io['stims'])
	stims = np.unique(astims)
	for s in stims:
		ids = np.nonzero(astims==s)[0]
		ss[s] = [io['evts'][e] for e in ids]
	return ss

def npres(cond):
	ss = stimsets(cond)
	return min([len(ss[k]) for k in ss])

def subsample(io, n):
	ss = stimsets(io)
	evts = []
	stims = []
	for s in ss:
		resps = np.random.permutation(len(ss[s]))
		for r in resps[:n]:
			stims.append(s)
			evts.append(ss[s][r])
	return gd.Doc({'stims':stims, 'evts':evts})

def srate(d):
	rate = float(len(flat(d['evts'])))/len(d['evts'])
	mrate = rate
	st = np.array(d['stims'])
	for s in np.unique(st):
		evts = [d['evts'][i] for i in np.nonzero(st==s)[0]]
		r = float(len(flat(evts)))/len(evts)
		mrate = max(r, mrate)
	return {'max_rate':mrate, 'mean_rate':rate}

def subsamp(d, n=4):
	nd = gd.Doc()
	conds = [k for k in d if k.startswith('cond')]
	for c in conds:
		sn = npres(d[c])
		nn = int(round(float(sn)/n))
		nd[c] = subsample(d[c], nn)
	return nd

def fusegrp(g):
	l = []
	for et in g:
		l.extend(et)
	l.sort()
	return tuple(l)

def groupevts(evts, n):
	inds = np.random.permutation(len(evts))
	i = 0
	g = []
	while i < len(evts):
		grp = [evts[gi] for gi in inds[i:i+n]]
		g.append(fusegrp(grp))
		i = i + n
	return g

def superpose(io, n, israte=False, ct=.3):
	if israte:
		rt = float(len(flat(io['evts'])))/len(io['evts'])
		n, rem = divmod(n/rt, 1)
		if rem>ct:
			n+=1
		if n <2:
			return io
	ss = stimsets(io)
	stims = []
	evts = []
	for s in ss:
		for g in groupevts(ss[s], n):
			stims.append(s)
			evts.append(g)
	return {'stims':stims, 'evts':evts, 'ncomb':n} 

def combineevts(c, rt = 'auto', eqn=True, ssctrl=False):
	conds = c.keys(0, 'cond')
	if rt == 'auto':
		rt = max([srate(c[con])['mean_rate'] for con in conds])
	nd = gd.Doc()
	for k in c:
		if k in conds:
			nd[k] = superpose(c[k], rt, True)
		else:
			nd[k] = c[k]
	if eqn:
		n = min([npres(nd[k]) for k in conds])
		for cond in conds:
			if ssctrl:
				nd[cond+'_ssctrl'] = subsample(c[cond], n)
			if npres(nd[k]) > n:
				nd[cond] = subsample(nd[cond], n)
	return nd	
	
class ECombine(Transform):
	defaults = {'cell':'->', 'rate':'auto', 'eq':True, 'n':1}	
	def run(self, pars, out, messages):
		for i in range(pars['n']):
			c = combineevts(pars['cell'], pars['rate'], pars['eq'], False)
			for k in c.keys(0, 'cond'):
				if pars['n'] == 1:
					out[k] = c[k]
				else:
					out[k+"_%i" % i] = c[k]

class Window(Transform):
	defaults = {'cell':'->', 'windows':((0, 10), ('-10', '0'))}
	def run(self, pars, out, messages):
		out.patch(window(pars['cell'], pars['windows']))
		
class Jitter(Transform):
	defaults = {'cell':'->', 'jit':5000, 'insert':0, 'n':5}
	def run(self, pars, out, messages):
		d = pars['cell']
		for c in d.keys(0, 'cond'):
			out[c] = jitter(d[c], pars['jit'], 
		                pars['insert'], pars['n'])

class DMat(Transform):
	defaults = {'io':'->', 'dmeth':'ed_ist', 'q':None, 'nulls':None}
	def run(self, pars, out, messages):
		lot = pars['io']['evts']
		out['dm'] = dist.dist(lot, pars['dmeth'], pars['q'], pars['nulls'])	

class GroupDM(Transform):
	defaults = {'dm':'->dm', 'clust':'tree', 'nclust':8, 
	            'cargs':None, 'stims':'->stims'}	
	def run(self, pars, out, messages):
		dm = np.array(pars['dm'])
		if pars['clust'] == 'confusion':
			out['cm'] = dist.confusion(pars['stims'], dm)
		elif pars['clust'] == 'ident':
			out['evts'] = identdthresh(dm, 0, False)
		elif pars['clust'] == 'threshold':
			tmode = pars['cargs.tmode'] or 'first'
			out['evts'] = identdthresh(dm, pars['nclust'], True, tmode)
		else:
			cargs = pars['cargs'] or {}
			out['evts'] = clustdm(pars['clust'], dm, pars['nclust'], cargs)

class DMTree(Transform):
	defaults = {'dm':'->dm'}
	def run(self, pars, out, messages):
		dm = np.array(pars['dm'])
		t =  clust.dtree(dm)
		out['tree'] = clust.tree2tup(t)

class CMI(Transform):
	defaults = {'tree':"->tree", 'stims':"->stims", 'nclust':2, 'shuff':5}
	def run(self, pars, out, messages):
		t = clust.tup2tree(pars['tree'])
		part = t.cut(min(pars['nclust'], len(t)))
		io = gd.Doc({'stims':pars['stims'], 'evts':part})
		if type(pars['shuff']) in [str, unicode]:
			mi, ie, oe = minf(io, pars['shuff'], True)
			mi = [mi]
		elif pars['shuff']:
			mi, ie, oe = minf_db(io, 'direct', ('shuffle', pars['shuff']))
		else:
			mi, ie, oe = minf(io, 'direct', True)
		out['stiment'] = ie
		out['respent'] = oe
		out['mi'] = mi
		
class Group(Transform):
	def run(self, pars, out, messages):
		grpargs = pars['grpargs'] or {}
		out['io'] = classifyresp(pars['io'], pars['grp'], pars['q'], **grpargs)
		
class MI(Transform):
	'''
	expects io: int to int, and mim
	'''
	defaults = {'io':'->io', 'mim':'direct', 'debias':()}
	def run(self, pars, out, messages):
		db = pars['debias']
		if db:
			mi, ie, oe = minf_db(pars['io'], pars['mim'], db)
		else:
			mi, ie, oe = minf(pars['io'], pars['mim'], True)
			mi = [mi]
		out['stiment'] = ie
		out['respent'] = oe
		out['mi'] = mi

class Colate(Transform):
	defaults = {'r':'->','doc':'->indoc', 'xvar':'nclust'}
	def run(self, pars, out, messages):
		d = pars['doc']
		xvar = pars['xvar']
		r = pars['r']
		out.patch( gd.Doc({'xvar':xvar, 'docsource':d['docsource']}))
		for cond in d.keys(0, 'cond'):
			out[cond] = {'nspikes':len(flat(d[cond]['evts'])), 
				     'npres':len(d[cond]['evts'])}
			sc = r.findall({'_params.io':'=->%s' % cond})
			if sc:
				out[cond]['params'] = r[sc[0]]['_params']
			m = {}
			for k in sc:
				x = r[k]['_params'][xvar]
				m[x] = r[k]['mi'] + [r[k]['stiment']]
			out[cond]['x'] = np.array(sorted(m))
			out[cond]['y'] = np.array([m[i] for i in out[cond]['x']])

win = Window()
jit = Jitter()
comb = ECombine()
dm = DMat()
grpdm = GroupDM()
grp = Group()
dmtree = DMTree()
cmi = CMI()
mi = MI()
colate=Colate()



