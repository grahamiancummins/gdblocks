#!/usr/bin/env python -3
# encoding: utf-8

#Created by gic on 
#Tue Nov 16 14:03:26 CST 2010

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

import numpy as np
from base import Optimizer, Model, Alg, ParamRange, Store


class GA_Prop(Alg):
	'''
	A genetic algorithm (more properly an Evolutionary Strategy, since it uses
	integer, rather than binary, chromosomes), which uses proportional
	selection, crossover, mutation, and transposition, worst-unit replacement,
	and no breeding history (no eletism, breeding delay, etc. Selfing is
	possible (indeed, it is probable if there is a single unit with a uniquely
	high fitness) 

	Parameters include: 
			mutation
			crossover
			transposition
			minprop
	
	'''

	def select(self, fit):
		fit = fit - fit.min() + self.pars['minprop']
		regions = np.cumsum(fit)
		picks = np.random.uniform(0, regions[-1], 2)
		return np.searchsorted(regions, picks)


	def mutate(self, c):
		n = np.random.randint(0, c.shape[0])
		v = np.random.randint(0, self.range.shape[n]+1)
		v = self.range.min[n]+v*self.range.step[n]
		c2 = c.copy()
		c2[n] = v
		return c2

	def cross(self, c1, c2):
		n = np.random.randint(0, c1.shape[0])
		c = np.concatenate( [c1[:n],c2[n:]] )
		return c

	def transpose(self, c):
		v1, v2 = np.random.permutation(c.shape[0])[:2]
		c2 = c.copy()
		c2[v1], c2[v2] = c[v2], c[v1]
		return c2

	def next(self, units):
		fit, chroms = self.arrays(units)
		dad, mom = self.select(fit)
		dad = chroms[dad,:]
		mom = chroms[mom,:]
		checks = np.random.uniform(0, 1, 4)
		if checks[0]< self.pars['mutation']:
			dad = self.mutate(dad)
		if checks[1]< self.pars['mutation']:
			mom = self.mutate(mom)
		if checks[2]< self.pars['crossover']:
			c = self.cross(dad, mom)
		else:
			c = dad
		if checks[3]<self.pars['transposition']:
			c = self.transpose(c)
		return tuple(c)

	def kill(self, units):
		z = units.items()
		f = np.array([i[1] for i in z])
		bi = f.argmin()
		u = z[bi][0]
		del(units[u])
		

def test():
	p = ParamRange(np.array([[-1,1, 10],[0,5, 10 ],[-2,1.1, 10] ]))
	s = Store('testga.giclog')
	pars = {'threads':0, 'size':20, 'time':0.1}
	model = Model() # test implementation that return product of params
	model.perfect = 10
	gapars = {'crossover':.9, 'mutation':.05,'transposition':.001, 'minprop':0 }
	alg = GA_Prop(gapars, p)
	o = Optimizer(pars, model, alg, s)
	o.run()



if __name__ == '__main__':
	test()


