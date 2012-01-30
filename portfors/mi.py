#!/usr/bin/env python -3
# encoding: utf-8

#Created by Graham Cummins on Jun 3, 2011

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

from gicdat.control import report
import numpy as np
from grouping import idents
from gicdat.util import maxdiag
import cext.portforsgic as pfce #@UnresolvedImport

try:
	from pyentropy import DiscreteSystem
except:
	report("Warning, no pyentropy. Better stick to the direct method!")
	
	
def _jprob(x, y, a):
	return float(np.logical_and(a[:,0]==x, a[:,1]==y).sum())/a.shape[0]

def h_direct(s):
	s = np.array(s)
	prob = np.bincount(s)/float(s.shape[0])
	h =  -1.0*np.array([prob[x]*np.log(prob[x]) for x in np.unique(s)]).sum()/np.log(2)
	return h

def minf_c_direct(cond):
	return pfce.mi_direct(cond['stims'], cond['evts'])

def minf_direct(cond):
	isp = np.array(cond['stims'])
	osp = np.array(cond['evts'])
	jsp = np.column_stack([isp, osp])
	iprob = np.bincount(isp)/float(isp.shape[0])
	oprob = np.bincount(osp)/float(osp.shape[0])
	tested = {}
	mi = 0.0
	hx = -1.0*np.array([iprob[x]*np.log(iprob[x]) for x in np.unique(isp)]).sum()/np.log(2)
	hy = -1.0*np.array([oprob[x]*np.log(oprob[x]) for x in np.unique(osp)]).sum()/np.log(2)
	for x, y in jsp:
		if x in tested:
			if y in tested[x]:
				continue
			else:
				tested[x].append(y)
		else:
			tested[x] = [y]
		jpr = _jprob(x,y,jsp) 
		la = jpr / (iprob[x]*oprob[y])
		mi += jpr * np.log(la)
	mi = mi/np.log(2)	
	return (mi, hx, hy)

def _pyentcond(l):
	a = np.array(l)
	a = np.reshape(a, (1,-1))
	a = a - a.min()
	s = (1, a.max()+1)
	return (a, s)

def minf_pyent(isp, osp, meth):
	x, xdim = _pyentcond(osp)
	y, ydim = _pyentcond(isp)
	ds = DiscreteSystem(x, xdim, y, ydim)
	calc = ["HX", "HXY", "HY"]
	if meth.endswith('_sh'):
		meth = meth[:-3]
		calc.extend(['HiXY', 'HshXY'])
	ds.calculate_entropies(meth, calc=calc)
	if len(calc)==3:
		return (ds.I(), ds.H['HY'], ds.H['HX'])
	else:
		return (ds.Ish(), ds.H['HY'], ds.H["HX"])
	
MIFUNCS = {
	'direct':minf_c_direct,
	'direct_py':minf_direct,
	}

def pyentFactory(s):
	def foo(cond):
		x = np.array(cond['stims'])
		y = np.array(cond['evts'])
		return minf_pyent(x,y,s)
	return foo
	
for s in ['plugin', 'pt', 'qe', 'nsb', 
		  'plugin_sh', 'pt_sh', 'qe_sh', 'nsb_sh']:
	MIFUNCS[s] = pyentFactory(s)


def cmatMI(cond):
	'''
	return the mutual information associated to a confusion matrix.
	'''
	c = cond['cm']
	pd = c/c.sum()
	rm = pd.sum(1)
	cm = pd.sum(0)
	mi = 0.0
	hx = -1.0* (rm * np.log(rm+(rm==0))).sum() / np.log(2)
	hy = -1.0* (cm * np.log(cm+(cm==0))).sum() / np.log(2)
	for i in range(c.shape[0]):
		for j in range(c.shape[0]):
			jpr = pd[i,j]
			if jpr == 0:
				continue
			la = jpr / (rm[i]*cm[j])
			mi += jpr * np.log(la)
	mi = mi/np.log(2)	
	return (mi, hx, hy)	
	
MIFUNCS['cmat'] = cmatMI



def _bl_d(a):
	dm = np.zeros((a.shape[0], a.shape[0]))
	for i in range(dm.shape[0]-1):
		for j in range(i+1, dm.shape[0]):
			d = np.sqrt( ((a[i,:] - a[j,:])**2).sum())
			dm[i,j] = d
			dm[j,i] = d
	return maxdiag(dm)

def binless(cond):
	scounts = np.array([len(t) for t in cond['evts']])
	hy = h_direct(idents(cond['evts']))
	mi_ct, hx, _ = minf_direct({'stims':cond['stims'], 'evts':scounts})
	mi_n = np.zeros(len(scounts))
	stimvals = np.unique(cond['stims'])
	stims = np.array(cond['stims'])
	for i, sc in enumerate(sorted(set(scounts))):
		if sc == 0:
			continue
		inds = np.nonzero(scounts == sc)[0]
		stsc = stims[inds]
		Nn = float(inds.shape[0])
		sprob = np.zeros(stimvals.shape[0])
		for k,s in enumerate(stimvals):
			Nak = (stsc == s).sum()
			if Nn < 2 or Nak < 2:
				continue
			sprob[k] = (Nak/Nn) * np.log2((Nak-1) / (Nn-1) )
		resps = np.array([  cond['evts'][i]  for i in inds])
		dists = _bl_d(resps)
		lam = np.zeros(dists.shape[0])
		lam_s = np.zeros(dists.shape[0])
		mav = dists.max() 
		dm = np.where(dists==0, mav, dists)
		miv = dm.min()
		dists = np.where(dists==0, miv, dists)
		for k in range(lam.shape[0]):
			lam[k] = dists[k,:].min()
			ss = np.array([ii for ii in range(lam.shape[0]) if stsc[ii] == stsc[k]])
			if ss.shape[0] < 2:
				lam_s[k] =mav
			else:
				lam_s[k] = dists[k,ss].min()
				if lam_s[k] == 0:
					lam[k] = 1
					lam_s[k] = 1
		mi_n[i] = (sc /Nn) * np.log2(lam/lam_s).sum() - sprob.sum() 
	mi = mi_ct + mi_n.sum()
	return (mi, hx, hy)

MIFUNCS['binless'] = binless

def minf(cond, method="direct", entr=False):
	mi = MIFUNCS[method](cond)
	if entr:
		return mi
	else:
		return mi[0]
	
def shuffle(cond):
	return {'stims':cond['stims'], 'evts':np.random.permutation(cond['evts'])}	
	
def minf_db(cond, method='direct', debias =('shuffle', 10)):
	mi, se, oe = MIFUNCS[method](cond)	
	if debias[0] == 'shuffle':
		miv = []
		for _ in range(debias[1]):
			sc = MIFUNCS[method](shuffle(cond))[0]
			miv.append(sc)
		miv = np.array(miv)
		se = se - miv.mean()
		miv = mi - miv
		mi = miv.mean()
		er = (miv.max()-mi, mi-miv.min())
	else:
		mima = mi
		mimi = mi
		for meth in debias:
			m = MIFUNCS[meth](cond)[0]
			mima = max(mima, m)
			mimi = min(mimi, m)
			er = (mima -mi, mi - mimi)
	return (mi, se, oe, er)
