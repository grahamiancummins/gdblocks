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
import numpy as np

"""
cd -> CellDoc
ce -> CellExp
cond -> IntIO


"""

def spikes_from(cond, stim):
	"""
	return a flat list of every spike in cond['evts'] resulting from stmulus 
	stim (int). 
	
	"""
	resps = [cond['evts'][i] for i in range(len(cond['evts'])) 
	         if cond['stims'][i] == stim]
	return flat(resps)


def response_densities(cond, ncent = (2, 8)):
	"""
	return a list l such that l[i] = is a gmm of the responses to stimulus i.
	
	Since these are 1D models, the underlying mixmod call always uses PkLkCk,
	and only one repeat, but ncent specifies the min and max values of the 
	number of centers to try.
	
	"""
	cents = apply(range, ncent)
	stims = set(cond['stims'])
	l = [None for _ in range(max(stims)+1)]
	for s in stims:
		sf = spikes_from(cond, s)
		if sf:
			l[s] = mmcall(np.array(sf)[:,np.newaxis], cents)	
	return l


