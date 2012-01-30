
#!/usr/bin/env python -3
# encoding: utf-8

#Created by gic on 
#Sun Nov 28 10:59:34 CST 2010

# Copyright (C) 2010 Graham I Cummins
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA 02111-1307 USA
#
from __future__ import print_function, unicode_literals
import acell, gwn
import ga as iga
import paper as ip
import gicdat.io as gio
import matplotlib.pyplot as plt
import numpy as np


def bestn(d, i):
	n = []
	f = []
	for k in d:
		if d[k]['fit']>0:
			n.append(k)
			f.append(d[k]['fit'])
	f = np.array(f)
	a = np.argsort(f)
	a = a[-i:]
	a = a[::-1]
	return [n[i] for i in a]

def nth(d, i, node = True):
	n = []
	f = []
	for k in d:
		n.append(k)
		f.append(d[k]['fit'])
	f = np.array(f)
	a = np.argsort(f)
	a = a[::-1]
	node = d[n[a[i]]]
	if node:
		return node
	else:
		return iga.parse(node['pars'])


def acc(**kw):
	defaults = {'nd':3, 'slen':800, 'compress':'istac', 'clevel':4}
	defaults.update(kw)
	model = iga.AcellCross(defaults)
	return model


def show(n, compress='istac', clevel=.85, nd=3):
	m1, m2 = iga.parse(n['pars'], 3)
	ip.bigFig(m1, m2, compress, clevel)

def stimlengthscan(n, mi=50, ma=3000, step=100, sd=5):
	fvs = []
	import rand
	rand.FROZEN = False
	for sl in range(mi, ma, step):
		ev = []
		for i in range(sd):
			ac = acc(slen=sl)
			ev.append(ac.eval(n['pars'])[0])
		fvs.append( (np.mean(ev), np.std(ev) ) )
		print(fvs[-1])
	rand.FROZEN = True
	return fvs