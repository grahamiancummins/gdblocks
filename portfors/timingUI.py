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
Functions for high level interface of the timing information project. 

This includes mostly scanning across a range of parameters to construct plots 
of mutual information as a function of various conditions relating to the 
temporal precision of responses, and displaying these plots.

Also included are some of the most recent functions for reading and organizing 
Batlab data

tags:

CondName from selectUI
DirName from selectUI
DistMode from selectUI
DistQ from selectUI
MIMode from mi
MIDebias from mi


EvtSet(N): N-[ of [ of i 
	This is a collection identical discreet events in
	time. Each element of self is a single response, R, and each element of R is
	the time stamp of an event in the sequence. The event times are represented
	as integer numbers of microseconds, and are relative to some reference time
	(such as the start of an recording session). The responses may have different
	lengths, including 0, and there is not a specific upper bound on the length. 
	The event indexes may be negative. Although raw recordings from Batlab 
	always yeild non-negative events, subsequent functions may realign the start
	time (for example to stimulus onset), allowing some events to occur at 
	negative times. 

IntIO: {evts:EvtSet(N), stims: N-[ of i}
	These documents represent an eperiment in which the inputs are enumerable 
	distinct stimulus conditions, and the outputs are EvtSets. If evts[i] = R
	and stims[i] = j, then R occured as a response to presentation of stimulus 
	number j. 

CellExp; {KeySet('cond', N):IntIO} :: N>=1 (*)
	
	(*) KeySet('cond', N) indicates a set of keys cond1, cond2 ... condN This is
	a common pattern, but not in the gicdat.search standard for tags yet.
	
	This is a collection of IntIO experiments conducted under differnt
	conditions (represented by the index number in the KeySet). In the Portfors
	IC experiments, cond1 is the control condition and cond2 includes the GABA
	blockers Bicuculine (BIC) and Strychnine (Strych). Some experiments also
	include conditions 3, 4, and 5, which are post-wash-out recovery, BIC only
	and Strych only (I think), but we aren't currently testing those. These
	CellExps also have additional keys docsource,
	
	maxstimdur, and stimclasses, which specify where the information was read
	from, what the longest stimulus window is, and the details of each of the
	enumerated stimuli used. The spec for these keys is not stable, and at least
	two are in use. I will try to standardize and document these eventually.

FilePath: s
	
	The name of a file in the filesystem. May be relative or absolute, but
	does not include the expansion of environment variables or "~"

TestInfo: t of tuple, such that apply(frompst, t) is meaningful. See the
	docstring for frompst
	
Scan: Doc

	The document structure resulting form scanning functions.
	
TranName(M): s

	The name of a Transform, defined in module M, without the module prefix,
	so if x: TranName('portfors.timingInfo') = "pproc" then it refers to 
	the Transform portfors.timingInfo.pproc. 

Server: s

	The url of a remote computation server, or the string "local", or False.
	"local" differs from False in that "local" uses the paralell distribution 
	system on the local host, while False runs the job single-threaded in the 
	current python process. False is thus preffered for debugging, and on 
	single core machines, but "local" should perform better (if the distributer
	is working) on multi-core machines.
	
Params(T): Doc

	A document providing the parameters for transform t: TranName. The keys
	depend on the transform
	
Result(T): Doc

	The document resulting from running a transform T: TranName. The keys depend
	on which transform is called. See the trasform documentation.
	
JobOut(N): {KeySet('j', N): D}
	The output of running a gicdat.jobs Job or job set. The number of keys 
	depend on how man processes were in the job. The content of the keys D
	depends on the job. Typically, this is some for of Result. For a simple
	job, all the D are a particular Result(T). This can be abreviated 
	JobOut(N, T).



(*) KeysOf(d, p?)  (add this to the search spec)
Indicates each key in d, so a type {KeysOf(d):x} is a document that has all the
same keys as document d, and each one keys a floating point number. If the 
optional parameter p is uses, it matches a prefix, so {KeysOf(d, "cond"):x} 
contains every key in d that starts with the string "cond".

"""


from __future__ import print_function, unicode_literals

import gicdat.doc as gd
import gicdat.io as gio
import gicdat.jobs as gdjob
import numpy as np
from gicdat.util import traverse
import matplotlib.pyplot as plt
import vis
import gicdat.stdblocks.seq as gdseq
from operator import add as _add_
from gicdat.control import report
import scipy.optimize as opt
from gicdat.enc import flat
import tests as T
import os  
from mpl_toolkits.mplot3d import Axes3D #@UnusedImport
import timingInfo as tin
from timingInfo import srate, npres, combineevts, subsamp
import re

MDIR = os.path.split(__file__)[0]
STIMDIR = os.path.join(MDIR, 'stims')
CDF = os.path.join(MDIR, 'cells.gic')
OUT = os.path.join(MDIR, 'measurements.csv')
QR= [1] + range(1000, 10000, 2000)
SERVER = None
BP="gdblocks.portfors.timingInfo."
PSTDIR = os.path.expanduser("~/project/christine/bicIC/pst")
vl = str('http://van-lorax.vancouver.wsu.edu/gd')
lh = str('http://localhost:4242')

TESTS = {
    5 : ("Mouse 735b.pst",("71-86","90-105"),  25, 5 ),  
    8 : ("Mouse 739.pst",("74-89","91-106"),  25, 8 ),
    14: ("Mouse 744a.pst",("3-18","20-35"), 	15, 14),
    15: ("Mouse 744a.pst",("37-52","71-86"), 	15, 15),
    16: ("Mouse 744a.pst",("94-109","112-127"), 	15, 16),
    19: ("Mouse 746a.pst",("94-108","110-124"), 	15, 19),
    20: ("Mouse 746a.pst",("46-60","62-76"), 	15, 20),
    32: ("Mouse 755a.pst",("37-52","54-69"), 	15, 32),
    33: ("Mouse 755a.pst",("138-153","172-187"), 	15, 33),
    35: ("Mouse 763a.pst",("2-17","19-34"), 	15, 35),
    39: ("Mouse 806.pst",("3-18","20-35"), 	25, 39),
    40: ("Mouse 806.pst",("40-55","57-72"), 	25, 40),
    42: ("Mouse 806a.pst",("21-36","54-69"), 	15, 42),
    44: ("Mouse 870a.pst",("48-63","82-97"), 	35, 44),
    45: ("Mouse 871.pst;Mouse 871a.pst",("96-111","18-33"), 	15, 45),
    48: ("Mouse 886a.pst",("26-41","84-99"), 	15, 48),
    49: ("Mouse 886a.pst",("117-132","167-182"), 	15, 49),
    50: ("Mouse 886a.pst",("207-222","224-239"), 	15, 50),
    51: ("Mouse 886b.pst",("40-55","82-97"), 	15, 51),
    52: ("Mouse 893o.pst", ('3-32', '34-63'), 15.0, 52),
    53: ("Mouse 943.pst", ('32-43','47-58'), 15,  53),
    54: ("Mouse 943.pst", ('126-142', '144-160'),  25, 54),
    55: ("Mouse 953a.pst", ('19-32', '34-47'), 15, 55),
    56: ("Mouse 958.pst", ('38-51', '52-65'), 15, 56),
    57: ("Mouse 958.pst", ('128-141', '142-155'), 15, 57),
    58: ("Mouse 958a.pst", ('35-48', '64-77'), 15, 58),
    59: ("Mouse 958b.pst", ('18-31', '46-59'), 'all', 59),
    60: ("Mouse 964.pst", ('27-49', '51-64'), 'all', 60)
    }

NOTES = """
52 has many stims (30) at atten 15

56 has one attenuation and only 14 stims?

57 has atten 25 only in condition 2. It is a good example of an onset/offset
cell, which would benefit from having stimuli of fixed length

58 has very many attenuations.

59 has many attenuations, but is not usable, because no single attenuation is 
presented in both conditions

The PST file associated to 60 is damaged and can't be parsed.


subsampling effects the slope of the dmtree graphs a lot. Fewer total
presentations per stimulus equal more information with fewer clusters

cells >=44 have larger amounts of data. 46 and 47 are condition 1 only

rate equalization is done by "combineevts", imported from timingInfo

"""

CONDS = ['cond1', 'cond2']
TEST = [5, 8,14,15,16, 19,20,32,33,35,39, 40, 42, 44, 45, 48, 
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
GOOD = [8, 32, 33, 35, 50, 51, 52, 55, 57, 58]

STIMDUR = 143.856

def seq_eval(scan, d):
	if SERVER == "TEST":
		return gd.Doc()
	elif SERVER:
		return gdjob.distribute_sequence(scan, d)
	else:
		return gdjob.do_sequence(scan, d)

def _nrs_rt(c):
	s = set(c['stims'])
	sr = set()
	for i, e in enumerate(c['evts']):
		if e:
			sr.add(c['stims'][i])
	return len(s.difference(sr))/float(len(s))
		
def _stiment(c):
	stims = []
	for i, e in enumerate(c['evts']):
		if e:
			stims.extend([c['stims'][i]]*len(e))
	isp = np.array(stims)
	iprob = np.bincount(isp)/float(isp.shape[0])
	hx = -1.0*np.array([iprob[x]*np.log(iprob[x]) for x in np.unique(isp)]).sum()/np.log(2)
	return hx

def _nullrate(c):
	npre = len(c['evts'])
	nnul = 0
	for e in  c['evts']:
		if not e:
			nnul+=1
	return float(nnul)/npre

CMEAS = {'max_spikes':lambda x:max([len(e) for e in x['evts']]), 
         'mean_spikes':lambda x:
            np.array([len(e) for e in x['evts']]).mean(),
         'nullrate':_nullrate,
         'nonresponse_rate':_nrs_rt,'stim_entropy':_stiment}

def _shift(evtl, v):
	nel = []
	for et in evtl:
		nel.append(tuple(np.array(et) - v))
	return tuple(nel)

def _dirprep(dn):
	dn = os.path.expanduser(dn)
	if not os.path.exists(dn):
		os.mkdir(dn)
	return dn

def showpstteststimtype(pst):
	sd = gio.read(pst)
	for tk in sd.keys(0, 'test'):
		for k in sd[tk].keys(0, 'trace'):
			stim = sd[tk][k]['stim.ch0']
			if stim and stim.get('type'):
				print("%s : %s" % (tk, stim['type']))
				break
		else:
			print("%s : No Ch.0 stim" % tk)

def readpst(pst, tests = ("all",), atten = 'all', cellid=None, 
            stims = None, stimid = ('type','file','frequency')):
	'''
	pst: FilePath, tests: [ of TestRange, atten: "all" | x, addtodb: i?
		stims: { -> CellDoc
	
	TestRange of "all"|s|False|[ of i  (see description below)
		
	Reads an experiment document directly from a pst file, and returns it. This 
	is the same type of document returned by "getcell" and "load", containing 
	keys for different conditions, each of which contain "stims" and "evts" 
	keys. 
	
	This function differs from getcell in that it does not require a particular
	layout of the file system or a data spreadsheet. Instead, the argument "pst"
	is the full path to a pst file, and the argument "tests" specifies which
	tests in this file to load into each condition. "tests" is a list (or
	tuple). The ith value in tests specifies which tests to load into condition
	"cond%i % (i+1)" (so the 0 index value becomes "cond1", etc). Each value may
	be one of: a string containing a dash, which specifies an inclusive range of
	tests (e.g. "4-10"), a list of integers, which specifies tests exactly (e.g.
	[4,5,7,10]), the string "all", which refers to every test in the PST file,
	or a False value. False values result in not tests being loaded, and are
	used to allow loading data into higher numbered conditions without filling
	every condition (for example, if data are available for conditions 1, 2, and
	5, tests could be a 5-list with indexes 2 and 3 = False).
	
	Stimulus identities (integers) are determined by assigning an id to each
	unque combination of the stimulus keys (from stimulus ch0) indicated is the
	parameter stimid. For example, add "attenuation" to this list to treat
	different values of the attenuation as different stimuli
	
	If the argument stims is a dictionary, then it is used to check for existing
	assignments of a a given combination, and updated in place to reflect these.
	This can be used to ensure the same stimulus ids are assigned across
	serveral loads. (Note that it is for this use that "attenuation" is not in 
	the default stimid.)
	
	If "atten" is "all" (default), then each attenuation will be assigned as a
	different stimulus. If "atten" is a number, then all attenuations other than
	this one are discarded. If "atten" is "best", then the document is loaded
	using "all" and then 'keepbestatten'is run on it, which discards all stimuli
	other than the attunation which produces the most average spikes in the
	control condition.
	
	The final document will contain a structure "stimclasses" that shows the raw
	properties of each distinct stimulus. 
	
	If addtodb is a true value, then it should be an integer, and the result of 
	loading the tests will be stored in the cells.gic file under the key 
	"cell%i" % addtodb
	
	NOTE: This function uses a different (more general, and better) structure
	for the "stimclasses" sub-document than "getcell". Most of the functions in
	this module that use "stimclasses" know how to use either form.
	
	'''
	if stims is None:
		stims ={}
	if ";" in pst:
		#handles the annoying batlab crash case. Currently cell45 only
		fns = pst.split(';')
		od = readpst(fns[0], tests[:1], atten, cellid, stims)
		for i in range(1, len(fns)):
			d = readpst(fns[i], (tests[i],), atten, cellid, stims)
			od['cond%i' % (i+1,)] = d['cond1']
		od['docsource.file'] = pst
		return od
	sd = gio.read(pst)
	conds = []
	for tl in tests:
		if not tl:
			conds.append(None)
			continue
		conds.append({'stims':[], 'evts':[]})
		if type(tl) in [str, unicode]:
			if tl =='all':
				tl = [int(tn[4:]) for tn in sd if tn.startswith('test')]
			else:
				tl = map(int, tl.split('-'))
				tl = range(tl[0], tl[1]+1)
		for tid in tl:
			trd = sd['test%i' % tid]
			for k in trd:
				if k.startswith('trace'):
					tra = trd[k]
					stim = tra['stim.ch0'] or {'type':'No Stimulus'}
					evts = _shift(tra['events'], 
					                      stim.get('delay', 0)*1000.0)
					if not atten in ['all', 'best']:
						if not stim.get('attenuation') in [None, atten]:
							continue
					sdesc = tuple([stim.get(stk) for stk in stimid])
					if not sdesc in stims:
						stims[sdesc] = (len(stims), stim)
					sids = [stims[sdesc][0]]*len(evts)
					conds[-1]['stims'].extend(sids)
					conds[-1]['evts'].extend(evts)
	od = gd.Doc()
	for i, c in enumerate(conds):
		if c:
			od['cond%i' % (i+1,)] = c
	msd = 0
	for s in stims:
		i, ss = stims[s]	
		od['stimclasses.stim%i' % i] = ss
		msd = max(msd, ss.get('duration',0))
	od['maxstimdur'] = msd*1000
	od['docsource'] = {'file':pst, 'tests':tests, 
	                   'atten':atten, 'cell':cellid}
	if atten == 'best':
		od = discardatten(od)
	return od
	
def rfromstim(d, cond, stim):
	'''
	d: CellExp, cond: CondName(d), stim: i -> resps: EvtSet
	
	resps is the set of events in d[cond] that occur in response to stimulus 
	number stim. 
	
	'''
	resps = [d[cond]['evts'][i] for i in range(len(d[cond]['evts'])) 
	         if d[cond]['stims'][i] == stim]
	return resps

def ratestats(resps):
	'''
	resps: EvtSet -> s: {s of x
	
	return a dictionary containing firing rate and distribution statistics for 
	the responses resps <[(int..)...]>. Read the code for the list of stats 
	and keys
	
	s has keys:
	avr (average rate), maxr (max rate), minr (min rate), rstd (std dev of rates)
	nzero (number of zero rate responses), total (number of total responses)
	first (earliest spike time), last (latest spike time), tstd (response time
	std dev), avt (mean spike time).
	
	Rates are expressed in spikes per response, so to get a rate in Hz, you 
	need to know the length of the window (and also that all the reported 
	responses have the same window)
	
	'''
	s = {}
	rts = np.array([len(x) for x in resps])
	sts = np.array(flat(resps))
	#rts = rates: # of spikes for each presentation
	#sts = spike times: time of every event, put together in a single 1D array
	s['avr'] = rts.mean()
	s['maxr'] = rts.max()
	s['minr'] = rts.min()
	s['rstd'] = rts.std()
	s['nzero'] = (rts == 0).sum()
	s['total'] = rts.sum()
	s['first'] = sts.min()
	s['last'] = sts.max()
	s['tstd'] = sts.std()
	s['avt'] = sts.mean()
	return s

def atteninfo(d):
	'''
	d: CellExp -> attens of {x of {s of x
	
	Return a dictionary sumarizing the reponse statistics for different 
	attenuations in d
	
	Each subdictionary of attens contains keys:
	
	sids: [ of i (which stimuli are played at this attenuation)
	nnrs: i  (number of non response stimuli at this attenuation)
	all keys from "ratestats"
	
	'''
	attens = {}
	for sk in d['stimclasses']:
		if not sk.startswith('stim'):
			continue
		sid = int(sk[4:])
		attn = d['stimclasses'][sk]['attenuation']
		if not attn in attens:
			attens[attn] = {'sids':[]}
		attens[attn]['sids'].append(sid)
	for attn in attens:
		atd = attens[attn]
		for k in d.keys():
			if not k.startswith("cond"):
				continue
			resps = []
			nnrs = 0
			for sid in atd['sids']:
				srs = rfromstim(d, k, sid)
				if not srs:
					continue
				if max([len(x) for x in srs]) == 0:
					nnrs+=1
				resps.extend(srs)
			if not resps:
				continue
			atd[k] = ratestats(resps)
			atd[k]['nnrs'] = nnrs
	return attens

def bestatten(d, mode='avr', cond='cond1', getmax=True):
	'''
	d: CellExp, mode: s, cond: CondName(d), getmax: t -> x
	
	Choose the attenuation that maximizes or minimizes one of the stats
	returned by atteninfo. mode is a key into the atteninfo dictionaries. "cond"
	specifies which condition to check the key in, and getmax specifies whether
	to pick the max or the min.
	
	'''
	ai = atteninfo(d)
	if len(ai) == 1:
		return ai.keys()[0]
	if getmax:
		mv = -np.inf
	else:
		mv = np.inf
	for attn in ai:
		v = ai[attn][cond][mode]
		if getmax:
			if v> mv:
				mv = v
				ba = attn
		else:
			if v<mv:
				mv = v
				ba = attn
	return ba

def discardatten(d, mode='avr',  cond='cond1',getmax = True,):
	'''
	d: CellExp, mode: s, cond: CondName(d), getmax: x -> CellExp
	
	Discard all stimuli other than those with the "best" attenatuation. This
	also renumbers the stimuli, so the retained versions will be numbered in 
	a continuous block starting with 0. The return value is a new document. The
	input document is not modified in place. 
	
	The "best" attenuation is calculated by bestatten(d, mode, cond, getmax)
	The default mode is the attenuation for which the average spike rate is 
	highest.
	
	'''
	attn = bestatten(d, mode, cond, getmax)
	nd = gd.Doc()
	for k in ['docsource', 'maxstimdur']:
		nd[k] = d[k]
	nd['stimclasses'] = {}
	strans = {}
	for sk in d['stimclasses']:
		if d['stimclasses'][sk]['attenuation'] == attn:
			sid = int(sk[4:])
			nsid = len(nd['stimclasses'])
			strans[sid] = nsid
			nd['stimclasses.stim%i' % nsid] = d['stimclasses'][sk]
	for k in d:
		if k.startswith('cond'):
			nst = []
			nevt = []
			for ei, sid in enumerate(d[k]['stims']):
				if sid in strans:
					nst.append(strans[sid])
					nevt.append(d[k]['evts'][ei])
			nd[k] = {'stims':nst, 'evts':nevt}
	return nd

def load(cell = 58, eqstim=True, pois=False, win=(0,0), 
         conds = CONDS, strip=False):
	"""
	cell: i -> CellExp
	
	obsolete. Equivalent to gio.read(CDF)["cell%i" % cell]. celldoc or readpst
	provide more options.
	
	"""
	return gio.read(CDF)["cell%i" % cell]

def presentedstims(cd, cond):
	sids = set(cd[cond]['stims'])
	ss = {}
	for sid in sids:
		ss[sid] = cd['stimclasses.stim%i.file' % sid].split('.')[0]
	return ss

def convert_stimno(cond, sord):
	"""internal for unifystims"""
	stims = []
	evts = []
	for i, si in enumerate(cond['stims']):
		try:
			nsi = sord.index(si)
			stims.append(nsi)
			evts.append(cond['evts'][i])
		except ValueError:
			pass
	return {'stims':stims, 'evts':evts}

def unifystims(cd):
	"""internal for celldoc"""
	stims = None
	cks = cd.keys(0, 'cell')
	for cell in cks:
		for cond in cd[cell].keys(0, 'cond'):
			sids = set(cd[cell][cond]['stims'])
			if stims is None:
				stims = sids
			else:
				stims = stims.intersection(sids)
	stims = sorted(stims)
	sclasses = {}
	for i, sid in enumerate(stims):
		sn = 'stim%i' % sid
		for cell in cks:
			if sn in cd[cell]['stimclasses']:
				sclasses['stim%i' % i] = cd[cell]['stimclasses'][sn]
				break
		else:
			raise IndexError('Missing stimulus description for %s' % sn)
	ncd = gd.Doc()
	mstdur = max([n['duration'] for n in sclasses.values()])
	for cell in cks:
		ncd[cell] = {'stimclasses':sclasses, 'maxstimdur':mstdur}
		ncd[cell]['docsource'] = cd[cell]['docsource']
		for cond  in ['cond1', 'cond2']:
			ncd[cell][cond] = convert_stimno(cd[cell][cond], stims)
	return ncd

def celldoc(cells = GOOD, unify=True, pst='', writedb=False):
	"""
	cells: [ of i, unify: t, pst: DirName?, writedb: t-> 
			{Keyset('cell'):CellExp}
	
	Returns a document of cells corresponding to the integers in cells. These 
	cells all use the same integer IDs for each stimuls (as determined by the
	default stimid list of readpst).
	
	If unify, then only stimuli presented in every condition in every cell are
	retained, and these are renumbered to be on a continuous range starting 
	with 0.
	
	If pst, then the data are read from raw pst files stored in this directory,
	using the metadata in TESTS. Otherwise they are read from the CDF stored
	file. Unify may not be needed in this case (depending on how the CDF was
	written)
	
	if writedb, then the data are written to the CDF file (you probably only
	want this if pst is specified)
	
	"""
	if pst:
		cd = gd.Doc()
		cwd = os.getcwd()
		os.chdir(os.path.expanduser(pst))
		stims = {}
		try:
			for c in cells:
				cd['cell%i' % c] = apply(readpst, TESTS[c]+(stims,))
		finally:
			os.chdir(cwd)
	else:
		d = gio.read(CDF)
		cd = gd.Doc()
		for c in cells:
			cn = 'cell%i' % c
			cd[cn] = d[cn]
	if unify:
		cd = unifystims(cd)
	if writedb:
		gio.write(cd, CDF)
	return cd

def _drops(c, sn):
	ind = [si for si, s in enumerate(c['stims']) if not s ==sn]
	return gd.Doc({'evts':[c['evts'][i] for i in ind],
                   'stims':[c['stims'][i] for i in ind]})

def dropstim(d, sn):
	"""
	d: CellExp, sn: i|s -> CellExp
	
	Generate a copy of d that contains no presentations of stimulus sn, in 
	any condition. Sn is either a stimulus identifier integer, or the name 
	of a stimulus file. In the latter case, leave out the extension (.call1).
	
	"""
	if type(sn)!=int:
		try:
			sn = d['stimclasses'].find({'file':'=%s.call1' % sn}).next()
			sn = int(sn[4:])
		except:
			raise KeyError('stimulus is not in stimclasses')
	return traverse(d, 'cond', _drops, [sn], {}, 'new')

def make_thumbnail(stfn, mode='spectrogram', **kwargs):
	try:
		data = gio.read(os.path.join(STIMDIR, stfn))['data']
		if data is None:
			raise IOError("Couldn't read any data for stimulus")
	except:
		print("No stimulus waveform for %s. Can't make thumbnail" % stfn)
		return None
	im = vis.make_thumbnail(data[:,0], mode, **kwargs)
	return im

def set_thumbnails(d, **kwargs):
	nd = gd.Doc()
	if 'im_cache' in kwargs:
		cache = kwargs['im_cache']
		del(kwargs['im_cache'])
	else:
		cache = {}
	for s in d['stimclasses'].keys(0, 'stim'):
		stn = d['stimclasses'][s]['file']
		if stn in cache:
			i = cache[stn]
		else:
			i = make_thumbnail(stn, **kwargs)
			cache[stn] = i
		nd['stimclasses.'+s+'.thumbnail'] = i
	return d.fuse(nd)

def showcell(d, fig=1, save=None, roi = None, trng=(0,200), alpha=.8):
	'''
	d: CellExp, fig: i, save: FileName? -> None (draws in MPL figure fig. Writes
		to file save, if save is specified)
	
	Make a raster plot of cell document d, using MPL figure fig. If save is a 
	string, save the figure to that file name. pass trng, or look at (0,200)
	
	'''
	conds = [k for k in d.keys() if k.startswith('cond')]
	stims = np.unique(np.concatenate([d[k+'.stims'] for k in conds]))
	if trng =='auto':
		evts = np.array(flat([d[c]['evts'] for c in d.keys(0, 'cond')]))
		trng = (0, evts.max()/1000.0+1)
	nc = len(conds)
	rg=vis.RasterGrid(1, nc, fig)
	for i, c in enumerate(conds):
		rg.titles[0][i] = c[4:]
	rg.clear()
	for s in stims:
		grid = []
		sdict = d['stimclasses.stim%i' % s] or {}
		for cv in conds:
			rast = [d[cv+'.evts'][i] for i in range(len(d[cv+'.evts'])) if d[cv+'.stims'][i] == s]
			grid.append(rast)
		hlt = {}
		sst = sdict.get('onset') or 0.0
		sd = sdict.get('duration') or 0.0
		sp = sst+sd
		if roi:
			hlt['roi'] = tin.expandwindows(roi, sst, sp, 1.0)
			hlt['alpha'] = alpha
		stn = sdict.get('file', '--').split('.')[0]
		if sdict.get('thumbnail'):
			hlt['thumbnail'] = sdict['thumbnail']
			hlt['tnbounds'] = (0, sd)
		rg.add([grid], s, hlt, stn)
	rg.tmin = trng[0]
	rg.tmax=trng[1]  
	rg.draw()
	if save:
		plt.savefig(save)

def window(d, windows=[(0, 10), ('-10', '0')]):
	'''
	d: CellExp, windows: [ of (i|s, i|s) -> nd: CellExp
	
	As the function timingInfo.window, or the Transform timingInfo.wind,
	except that this function also corrects the stimuli in d['stimclasses'],
	by adjusting their onset, duration, and any thumbnails that exist for
	them already in a way appropriate to the window. 
	
	Note that "set_thumbnails" doesn't know about windows, so if you want to 
	window the thumbnails correctly, call this function first. Also, if you 
	only want to see the windows highlighted, don't use this function. Instead,
	pass the same "windows" argument as the "roi" argument of showcell.
	
	'''
	nd = d.fuse(tin.win(d, {'cell':'->', 'windows':windows})[0])
	expwin = tin.expandallwin(d, windows, 1.0)
	for s in nd['stimclasses'].keys(0, 'stim'):
		w = expwin[int(s[4:])]
		totlen = reduce(_add_, [x[1]-x[0] for x in w])
		sd = nd['stimclasses'][s]
		odur = sd['duration']
		sd['duration'] = totlen
		if w[0][0]:
			sd['onset'] = -w[0][0]
		if sd['thumbnail']:
			sd['thumbnail'] = vis.xwin_img(sd['thumbnail'], (0, odur), w)
	return nd

def combine(d):
	"""
	simple wrapper for tin.comb (n=1). Superposes responses in the sparser 
	conditions in CellExp d to approximate the rate in the louder conditions
	"""
	return d.fuse(tin.comb(d, {'cell':'->', 'n':1})[0])

def jitter(d, n=5, std=5000):
	"""
	Simple wrapper for tin.jit. Replace each response with n responses generated
	by applying gaussian noise with standard deviation std (and mean 0)
	"""
	return d.fuse(tin.jit(d, {'n':n, 'jit':std})[0]) 

def stim_srates(d):
	nd = gd.Doc()
	def ssrate(c, i):
		np = 0
		ns = 0
		for ei, e in enumerate(c['evts']):
			if c['stims'][ei] ==i:
				np+=1
				ns+=len(e)
		return float(ns)/np
	for k in d.keys(0, 'cond'):
		nd[k] = [ssrate(d[k], i) for i in sorted(set(d[k]['stims']))]
	return nd

def counting_measures(cells=TEST, conds=['cond1', 'cond2'], save=True):
	"""
	cells: [ of i, conds: [ of s, save: t -> Doc
	
	creates a spreadsheet of measurements performed on each cell in 
	celldoc(cells), and each condition in conds. If save is True, writes 
	this document to a csv file OUT. Returns the document, which has a 
	toplevel key for each cell, a subkey for each conditions, and subkeys
	within the conditions for each measurement. 
	
	The measurements are determined by the module dictionary CMEAS: {s of 
	def. Each function operates on an IntIO and results in a number or string
	"""
	cd = celldoc(cells)
	md = gd.Doc()
	for c in cd:
		for cond in conds:
			if cond in cd[c]:
				for m in CMEAS:
					md[c+'.'+cond+'.'+m] = CMEAS[m](cd[c][cond])
	if save:
		gio.write(md, OUT)
	return md

def scan_yval(s, x, cond):
	'''
	s: Scan, x: xi, cond: CondName(s)-> x
	
	return the y value associated to the trace for condition "cond" in scan s,
	where the value of _xvar is "x". Will throw an IndexError if the x coord
	doesn't take that exact value (e.g. no interpolation is performed).

	'''
	xi = np.nonzero(s[cond]['x'] == x)
	return s[cond]['y'][xi,0]
	
def markupscan(r, s, d, xvar='q'):
	'''
	Internal function used by scan functions to set some metadata
	'''
	if SERVER == "TEST":
		return s
	gdjob.annotate_seq(r,s, False)
	sd = gd.Doc({'xvar':xvar, 'docsource':d['docsource']})
	for cond in d.keys(0, 'cond'):
		sd[cond] = {'nspikes':len(flat(d[cond]['evts'])), 
		         'npres':len(d[cond]['evts'])}
		sc = r.findall({'_params.io':'=->%s' % cond})
		if sc:
			sd[cond]['params'] = r[sc[0]]['_params']
		m = {}
		for k in sc:
			x = r[k]['_params'][xvar]
			m[x] = [r[k][v] for v in ['mi', 'stiment']]
			if r[k]['estrange']:
				m[x].extend(list(r[k]['estrange']))
			else:
				m[x].extend([0.0, 0.0])
		sd[cond]['x'] = np.array(sorted(m))
		sd[cond]['y'] = np.array([m[i] for i in sd[cond]['x']])
	return sd

def estMI(cell, combine= 0, jit=0, jitsd=5000, win=( (0, STIMDUR),),
          dmeth='ed_ist', q=None, nclust=8,
          mim = 'direct', midebias = ('shuffle', 5)):
	'''
	cell: CellExp, combine: i, jit: i, jitsd:x, win:Windows,
		dmeth:DistMode, q:DistQ, nclust:i, mim: MIMode, 
		midebias:MIDebias -> ret: JobOut(1, "mi")
	
	Single-pass transform workflow for estimation of mutual information. 
	
	The long parameter list is passed to various of the transforms 
	See these transforms (in timingInfo) or the scan function for
	more detail.
	
	ret['j1'] is the result of a timingInfo.mi call. The mutual information
	itself is ret['j1.mi']
	
	'''
	js = []
	st = -1
	pr = []
	ret =[]
	if combine:
		js.append(gdjob.Job(BP+'comb', {'n':combine}))
	if jit:
		js.append(gdjob.Job(BP+'jit', {'n':jit, 'jit':jitsd}))
	if win:
		js.append(gdjob.Job(BP+'win', {'windows':win }))
	if js:
		st = len(js)-1
		pr = [-1] + range(st)
	for c in ['cond1', 'cond2']:	
		pr.append(st)
		js.append(gdjob.Job(
		    BP+'dm', {'io':'->%s' % c, 'dmeth':dmeth, 'q':q}))
		pr.append(len(js)-1)
		js.append(gdjob.Job(
		    BP+'grpdm', {'nclust':nclust, 'stims':'->'},
		    outpath = c))
		#don't need to specify stims for the default "tree" method, setting it to
		#root supresses a warning (about the path it is set to not being found)
		pr.append([st, len(js)-1])
		js.append(gdjob.Job(
		    BP+'mi', {'io':'->%s' % c, 'mim':mim,
		              'debias':midebias}))
		ret.append(len(js)-1)
	#ret = range(len(js))
	return seq_eval((js, gdjob.prlists(pr), ret), cell)

def scan_with_colate(scan, d, xvar='nclust'):
	scan = gdseq.seq2doc(scan)
	scan['colate.annotate'] = 0
	scan['colate.indoc'] = 'indoc'
	scan['colate.xform'] = BP+'colate'
	scan['colate.pars'] = {'xvar':xvar}
	if d:
		return gdjob.Job('stdblocks.seq.sequence', scan)(d)
	else:
		return scan	

def jit_scan(d, jit=QR, n=5, 
             dmeth='ed_ist', q=None, nclust=55, shuff=5):
	"""
	d: CellExp, jit: [ of x, n: i, windows: WinSpec, dmeth:DistMode, 
		q: DistQ, nclust: i, shuff: i ->
		Scan
	
	Scan interface for scanning across values of added jitter. jit specifies a
	range of jitter standard deviations to test. These values, and n are passed
	to tin.jit to construct jittered documents (n is the number of jittered
	repeats to use). dmeth and q are passed to tin.dm to control caluculation of
	a distance matrix. This is grouped by tree clustering into nclust clusters
	(by tin.grpdm), and MI is evaluated using tin.mi with debias parameter
	('shuffle', shuff).
	
	"""
	conds = ["->cond1", "->cond2"]
	tfs = [BP + tn for tn in ['jit','dm','grpdm','mi']]
	rs = [{'jit':jit},{'io':conds}, {}, {}]
	pars = [{'n':n},{'dmeth':dmeth, 'q':q},
	        {'nclust':nclust, 'stims':''},{'debias':('shuffle',shuff)}]
	forward = {2:{'io':(1, 'io', lambda x:'__'+x[2:])}, 
	           3:{'io':(1, 'io','')}}
	out = {2:'->io'}
	reentry = {3:[0]}
	scan = gdjob.tieredscan(tfs, pars, rs, out, forward, reentry)
	return scan_with_colate(scan, d, 'jit')

def win_jit_scan(d, jit=QR, n=3, windows=((0, STIMDUR),),
             dmeth='ed_ist', q=None, nclust=55, shuff=3):
	"""
	as jit_scan, with windows: WinSpec
	
	
	Very similar to jit_scan, except that between the transforms jit and dm, 
	also runs the transform tin.win with parameter "windows" to window the 
	responses.
	
	"""
	conds = ["->cond1", "->cond2"]
	#conds = ["->"+k for k in d.keys(0, 'cond')] 
	tfs = [BP + tn for tn in ['jit','win', 'dm','grpdm','mi']]
	rs = [{'jit':jit},{}, {'io':conds}, {}, {}]
	pars = [{'n':n},{'windows':windows}, {'dmeth':dmeth, 'q':q},
	        {'nclust':nclust, 'stims':''},{'debias':('shuffle',shuff)}]
	forward = {3:{'io':(2, 'io', lambda x:'__'+x[2:])}, 
	           4:{'io':(2, 'io','')}}
	out = {3:'->io'}
	reentry = {1:[-1], 4:[0, 1]}
	scan = gdjob.tieredscan(tfs, pars, rs, out, forward, reentry)
	return scan_with_colate(scan, d, 'jit')

def q_scan(d, qr=QR, dmeth='ed_bin', nclust=55, shuff=5):
	"""
	d: CellExp, qr: [ of DistQ, dmeth:DistMode, shuff:i, 
		mclust: i, serv:Server ->
		Scan
	
	Scan interface for scanning across a value of the parameter q in a distance
	measure. The scan uses dm to make a distance matrix with the chosen value of
	q, and then uses dmtree to cluster (with argument nclust), and mi to
	calculate mutual information (using direct ('shuffle', shuff))

	"""
	conds = ["->cond1", "->cond2"]
	tfs = [BP + tn for tn in ['dm','grpdm','mi']]
	rs = [{'io':conds, 'q':qr}, {}, {}]
	pars = [{'dmeth':dmeth},{'nclust':nclust, 'stims':''},
	        {'debias':('shuffle',shuff)}]
	forward = {1:{'io':(0, 'io', lambda x:'__'+x[2:])}, 
	           2:{'io':(0, 'io','')}}
	out = {1:'->io'}
	reentry = {2:[-1]}
	scan = gdjob.tieredscan(tfs, pars, rs, out, forward, reentry)
	return scan_with_colate(scan, d, 'jit')

def clust_scan(d, dmeth='ed_ist', shuff=5, mclust = 200, q=None):
	'''
	d: CellExp, dmeth:DistMode, shuff:i, mclust: i, q:DistQ, serv:Server ->
		Scan
	
	Restricted scan interface for scanning across a number of clusters if the
	clustering mode is "tree". This is faster than scan, for this case, since it
	only builds the tree one time, and slices it many times. Calls dm, dmtree, 
	cmi
	
	Mutual information is always direct with shuffle. The parameter shuff
	determines how many shuffles to do. 
	
	mclust is the maximum number of clusters to scan to. If False, tries all 
	separations of the Tree (e.g. sets mclust to the number of elements)
	
	dmeth and q are used by timingInfo.dm
	
	There are no preprocessing steps, so if you would like to superpose, window
	or jitter, use combine, window, or jitter functions by hand before the 
	scan.
	
	'''
	conds = ["->cond1", "->cond2"]
	#mclust = min([mclust]+ [len(d[k[2:]+'.evts']) for k in conds])
	tfs = [BP + tn for tn in ['dm','dmtree','cmi']]
	rs = [{'io':conds}, {}, {'nclust':range(2,mclust)}]
	pars = [{'dmeth':dmeth, 'q':q},{},{'shuff':shuff}]
	forward = {2:{'stims':(0, 'io','.stims')}}
	reentry = {2:[-1]}
	scan = gdjob.tieredscan(tfs, pars, rs, None, forward, reentry)
	return scan_with_colate(scan, d, 'nclust')

##Scan output commands
def bsep(s, cp = 'cond1', cn = None):
	'''
	s: Scan, cp:CondName(s), cn CondName(s), -> (x, x) 
	
	calculate the value of the xvar for which the difference of scan
	results for condition cp minus condition cn is greatest in scan s. If 
	cn is None, use the absolute value of cp
	
	returns (the x value, the value of the difference found there)
	
	'''
	mi = s[cp]['y'][:,0]
	if cn:
		mi = mi - s[cn]['y'][:,0]
	ms = np.argmax(mi)
	return (s[cp]['x'][ms], mi[ms])

def fusescans(los, names):
	'''
	los: [ of Scan, names: [ of s -> Scan
	
	Combine a list of scan documents (los <[Doc]>), by renaming each condition
	in the ith document to <original_name>_<names[i]>
	'''
	nd = gd.Doc()
	nd['docsource.cell'] = '? (several)'
	nd['xvar'] = los[0]['xvar']
	for i, s in enumerate(los):
		for c in s.keys(0, 'cond'):
			nd[c+"_"+names[i]] = s[c]
	return nd

def _errbar_scan(sk, normalize=False, perspike = False, **kwargs):
	xvals = sk['x']
	mi = sk['y'][:,0]
	err = sk['y'][:,2:]
	if normalize:
		mi = mi/yvals[:,1]
	if perspike:
		nn = sk['nspikes']/float(sk['npres'])
		mi = mi/nn
		err = err/nn
	plt.errorbar(xvals, mi, yerr=(err[:,0], err[:,1]), **kwargs)
	return mi

def _mark_bsep(s):
	plt.axvline(s['cond1.best_nclust'], color='b', linewidth=1)
	plt.axvline(s['cond2.best_nclust'], color='r', linewidth=1)
	plt.axvline(s['best_nclust'], color='k', linewidth=1)

def showscan(d, normalize=False, perspike=False, fname='', fig=1, 
             efit=False, tpars=None):
	'''
	d: Scan, normalize: t, perspike: t, fname: s, fig: i, efit:t ->
		None (draws in MPL figure fig. May write a file fname)
	
	Graphical display function to display scan documents. d is a scan document.
	It needs to have markup. Normalize and perspike are boolean constants that 
	determine if MI should be normalized (or shown per spike). If fname is as 
	non-empty string, save an image to that file name. 'fig' <int> specifies 
	which MPL figure to use. efit <bool> specifies whether to construct and show
	exponential fits to the data.
	'''
	styles = [('b', 's'), ('r','p'), ('g', 'h'), ('k', 'o'), ((0,.9,.9), 'd'), ((.9, 0,.9), 'H'), ((.9, .9, 0), '*'), ((.5,.5,.5), '1')]
	f = plt.figure(fig)
	plt.clf()
	xvar = d['xvar']
	for i,k in enumerate(d.keys(0, 'cond', sort=True)):
		sid = i % len(styles)
		z = _errbar_scan(d[k], normalize, perspike, color = styles[sid][0],
		                 marker=styles[sid][1], linewidth=3, label=k)
	if d.get('best_nclust'):
		_mark_bsep(d)
	plt.legend(loc='best')
	mid = 'MI'
	if normalize:
		mid = "Normed " + mid
	if perspike:
		mid = mid+"/spike"
	ds = "cell %s" % d['docsource.cell']
	pts = "%s: %s vs %s" % (ds, mid, xvar)
	if tpars:
		pts += "\n" + ','.join(["%s:%s" % (k, tpars[k]) for k in tpars])
	plt.title(pts)
	if xvar == 'nclust':
		x = d.find({'x':'#'}).next()
		plt.xlim([d[x]['x'].max(),d[x]['x'].min()])
	f.canvas.draw()
	if fname:
		plt.savefig(fname)
	return None

def showManyScans(sd, perspike=False, diffs = False):
	"""
	sd: {KeySet('cell', N):Scan}::N<=11, perspike: t -> 
		None (draws in mpl figure 1)
	
	Plot scans for each scan in sd, a document which has keys indicating a cell
	identity ('cell1', etc), refferencing a Scan document. Each cell is plotted
	in a particular color, with condition 1 in solid, and condition 2 in dashed.
	
	The color list is currently explicit, which limits the number of cells that 
	can be plotted to 11.
	
	If perspike is true, the mutual information per spike is plotted. Otherwise
	the total mutual information is plotted.
	
	"""
	f = plt.figure(1)
	plt.clf()
	cols =['b', 'g', 'r', 'k', 'c','m', 'y', 
	       'darkgreen', 'coral', 'brown', 'darkslateblue']
	for ci, c in enumerate(sd.keys(0, 'cell', sort=True)):
		if diffs:
			y1 = sd[c]['cond1.y'][:,0]
			y2 = sd[c]['cond2.y'][:,0]
			if perspike:
				y2 = y2/sd[c]['cond2.nspikes']
				y1 = y1/sd[c]['cond1.nspikes']
			plt.plot(sd[c]['cond1.x'], y2-y1, color=cols[ci],
				         linewidth=3, linestyle='solid', label=c+" 2-1")
			pass
		else:
			z = _errbar_scan(sd[c]['cond1'], False, perspike, 
		                 color = cols[ci], linewidth=2, linestyle='solid',
		                 label = c+'1')
			z = _errbar_scan(sd[c]['cond2'], False, perspike, 
		                 color = cols[ci], linewidth=3, 
		                 alpha=.7, linestyle='dashed',
		                 label = c+'2')
	plt.legend(loc='best')
	f.canvas.draw()	

def showManyScansSP(sd, perspike=False):	
	"""
	sd: {KeySet('cell', N):Scan}::N<=12, perspike: t -> 
		None (draws in mpl figure 1)

	Plot scans for each scan in sd, a document which has keys indicating a cell
	identity ('cell1', etc), refferencing a Scan document. Each cell is plotted
	in a separate subplot, with condition 1 in blue, and 2 in red. Pearson's R
	values are displayed on each plot.
			
	The plots are displayed is a 4x3 grid, which limits the number of cells that 
	can be usefully displayed to 12.
	
	If perspike is true, the mutual information per spike is plotted. Otherwise
	the total mutual information is plotted.	
	"""	
	f = plt.figure(1)
	plt.clf()
	mim = np.inf
	mam = 0
	for ci, c in enumerate(sd.keys(0, 'cell', sort=True)):
		sp=plt.subplot(4, 3, ci+1)
		z = _errbar_scan(sd[c]['cond1'], False, perspike, 
		                 color = 'b', linewidth=4, label ='cond1')
		mim = min(mim, z.min())
		mam = max(mam, z.max())
		z2 = _errbar_scan(sd[c]['cond2'], False, perspike, 
		                 color = 'r', linewidth=4, label ='cond2')
		r=_corcov(z, z2)
		plt.text(.8, .9, "R=%.3g" % r, transform=f.get_axes()[ci].transAxes)
		mim = min(mim, z2.min())
		mam = max(mam, z2.max())
		plt.title(c)
	for i in range(1, 13):
		sp = plt.subplot(4, 3, i)
		if not sp.is_first_col():
			sp.yaxis.set_visible(False)
			sp.xaxis.set_visible(False)	
		elif not sp.is_last_row():
			sp.xaxis.set_visible(False)	
		plt.ylim([mim-.05, mam+.05])
	f.subplots_adjust(left=.04, right=.99, bottom=.03, 
	                  top = .93, wspace=.03, hspace=.25)
	f.canvas.draw()		 

def _corcov(r1, r2):
	r1 = r1-r1.mean()
	r2 = r2-r2.mean()
	rr = r1*r2
	return rr.sum()/(np.sqrt( (r1**2).sum()) *np.sqrt((r2**2).sum()))
	
## Scanning Macro Functions
def _measure_sep(cd):
	conds = cd.keys(0, 'cond')
	for c in conds:
		if cd[c+'_hp']:
			t = bsep(cd, c, c+'_hp')
		else:
			t = bsep(cd, c, None)
		cd[c+'.best_MI']=t[1]
		cd[c+'.best_nclust']=int(t[0])
	avbnc = np.array([cd[c+'.best_nclust'] for c in conds]).sum()
	cd['best_nclust'] = int(round(float(avbnc)/len(conds)))
	return cd
	
def treebsep(cell, dmeth='ed_ist', shuff=5, mclust=80, 
             q= None, winlen=200):
	'''
	cell: CellExp, dmeth:DistMode, q: DistQ, shuff:i, winlen: x 
		mclust: i->  
		d: 
		{scan: Scan, KeysOf(cell, 'cond'): D}
		where D: {best_MI_sep: x (a value of mutual information difference)
				  best_MI_nclust: i (a value of nclust giving best separation)
				  }
	
	Macro function which runs a clust_scan and then a bsep. Most arguments are
	as for clust_scan.
	
	The main use of the return value is to extract the raw scan of dmtree
	(from the "scan" key), or the optimal values of nclust for each condition
	from d[cond]["best_MI_nclust"]
	
	If shuff is 0, this function adds in poisson models, and checks bsep as the
	difference between data and poisson. If shuff is positive, instead this 
	uses ('shuffle', shuff) difference debiasing, and then simply takes the 
	max debiased MI. If the Poisson models are calculated, the winlen argument
	is needed to calculate spike rates and ranges for the models.
	
	'''
	conds = cell.keys(0, 'cond')
	if not shuff:
		cell = T.addpois(cell, conds, winlen)
	s = clust_scan(cell, dmeth, shuff, mclust, q)
	return _measure_sep(s)

def wholescan(cell, jrange=QR, dmeth='ed_ist', q=None, mclust=100,
               shuff=5, windows=(), njit=5):
	'''
	cell: CellExp, jrange:[ of x, dmeth: DistMode, q:DistQ, mclust:i,
		shuff:i, windows:WinSpec -> Scan
	
	This is a macro to run the 2-scan cluster and jitter protocol.

	runs treebsep, followed by win_jit_scan (or jit_scan if windows is False).
	Arguments are as appropriate to those functions. Return is the result of the
	jit_scan.
	
	'''
	if windows:
		cw = window(cell, windows)
		winlen = cw['stimclasses.stim0.duration']
	else:
		cw = cell
		winlen = 200.0
	d = treebsep(cw, dmeth, shuff, mclust, q, winlen)
	nc = d['best_nclust']
	print('Using %i clusters' % nc)
	if windows:	
		s = win_jit_scan(cell, jrange,njit, windows, dmeth, q, nc, shuff)
	else:
		s = jit_scan(cell, jrange, njit, dmeth, q, nc, shuff)
	return s

def preproc(cells, tn= False, uni=True, drop=(), jitn=0, jitsd=5000, 
            comb=False, windows=()):
	"""
	macro that loads celldoc(cells, uni), and then runs dropstim, jitter,
	combine, window (in that order) on each cell in the document. Dropstim runs
	once for each stim value in the list drop. jitter runs with n=jitn, std =
	jitsd, comb runs at all if comb is True, and window runs with the windows
	argument
	
	If any of drop, jitn, comb, windows are False values, the associated
	function is not run (so with all default args, this function is equivalent
	to celldoc(cells)).
	
	f tn is True, run set_all_thumbnails. If it is a dictionary, pass it
	as keyword arguments. This occurs before windowing, to preserve the 
	thumbnails.
	
	"""
	cells = celldoc(cells, uni)
	if tn:
		if not isinstance(tn, dict):
			tn = {}
		set_all_thumbnails(cells, **tn)
	def _pp(cell, drop=drop, jitn=jitn, jitsd=jitsd, comb=comb,
	        windows = windows):	
		for k in drop:
			cell = dropstim(cell, k)
		if jitn:
			cell = jitter(cell, jitn, jitsd)
		if comb:
			cell = combine(cell)
		if windows:
			cell = window(cell, windows)
		return cell
	return traverse(cells, 'cell', _pp, [], {}, 'new')


## Functions to run procedures on all data. All of these expect a CellDoc as 
## first argument. That is, the return value of celldoc or preproc

def set_all_thumbnails(cd, **kwargs):
	"""
	Sets thumbnails on every cell in cd. Mostly just a demo of using traverse,
	but it also creates a common cache, which allows set_thumbnails to run 
	faster than if it is called separately for each cell.

	(really, cell docs should have a single stimclasses element anyway, and 
	individual CellExp should have references, but the reference system isn't 
	tested well enough to deploy right now)
	
	"""
	kwargs['im_cache']={}
	traverse(cd, 'cell', set_thumbnails, [], kwargs, 'mod')

def allrasters(cells, save="~/scratch", **kwargs):
	""" For every cell in cells, showcell(cell), saving the image as cell#.png
	in directory "save". kwargs are passed to showcell.
	
	"""
	save = _dirprep(save)
	for cell in cells.keys(0, 'cell'):
		fn = os.path.join(save, "%s.png" % cell)
		showcell(cells[cell], save=fn, **kwargs)

def allcscans(cells, save='',cpus=1, **kwargs):
	"""
	run treebsep (and thus clust_scan) for each cell in cells. kwargs are 
	passed to treebsep. 
	
	If save is a DirName, save an image of each scan with file name cell#.png
	in that directory. 
	
	Return the results ({KeySet('cell'):Scan)
	
	"""
	if cpus == 1:
		nd = traverse(cells, 'cell', treebsep, (), kwargs, 'patch')
	else:
		s = clust_scan(None, **kwargs)
		nd = gdjob.jobpool('stdblocks.seq.sequence', s, cells, 'cell', cpus)
		nd = traverse(nd, 'cell', _measure_sep, (), {}, 'patch')
	if save:
		save = _dirprep(save)
		for cell in nd.keys(0, 'cell'):
			fn = os.path.join(save, "%s.png" % cell)
			showscan(nd[cell], fname=fn)
	return nd

def alljscans(cells, save='', nc=None, cpus=1, **kwargs):
	if not nc:
		kw = {'dmeth':'ed_ist', 'shuff':5, 'q':None}
		for k in kw:
			if kwargs.get(k):
				kw[k] = kwargs[k]
		if kwargs.get('windows'):
			nc = traverse(cells, 'cell', window, 
			                 (kwargs['windows'],), {}, 'new')
			nc = allcscans(nc, '', cpus, **kw) 
		else:	
			nc = allcscans(cells, '', cpus, **kw) 
	sd = gd.Doc()
	for c in cells:
		ncl = nc[c]['best_nclust']
		print("%s (%i clusters)" % (c, ncl))
		if kwargs.get('windows'):	
			sd[c] = win_jit_scan(cells[c], nclust=ncl, **kwargs)
		else:
			sd[c] = jit_scan(cells[c], nclust=ncl, **kwargs)
	if save:
		save = _dirprep(save)
		for cell in sd.keys(0, 'cell'):
			fn = os.path.join(save, "%s.png" % cell)
			showscan(sd[cell], fname=fn)
	return sd



def show_win_bsep(wscan):
	bsv_c1 = []
	bsv_c2 = []
	bsv = {}
	sk = wscan.keys(0, 'w', sort=True)
	f = plt.figure(1)
	plt.clf()
	for i, k in enumerate(sk):
		bsv[k]  = [[], []]
		for c in wscan[k].keys(0, 'cell'):
			_measure_sep(wscan[k][c])
			bsv[k][0].append(wscan[k][c]['cond1.best_MI'])
			bsv[k][1].append(wscan[k][c]['cond2.best_MI'])
		plt.plot([i]*len(bsv[k][0]), bsv[k][0], 
		         linestyle='none', marker='o', color='b')
		plt.plot([i+.2]*len(bsv[k][1]), bsv[k][1], 
		         linestyle='none', marker='o', color='r')
	plt.xlim([-1, len(sk)])
	plt.xticks(range(0, len(sk)),
	           [str(wscan[k]['window']) for k in sk])
	
	f.canvas.draw()		