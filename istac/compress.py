#!/usr/bin/env python -3
# encoding: utf-8

#Created by Graham Cummins on 
#Mon Mar 21 11:41:36 CDT 2011

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
import numpy.linalg as la
import scipy.linalg as sla

def _5sf(f):
	''' return a 5 significant figure representation of number f '''
	return float("%.5g" % (f,))

def P(v):
	''' dot(v, v.transpose()) '''
	return np.dot(v, v.transpose())

def _norm(v):
	''' the 2-norm of vector v '''
	return float(np.sqrt( (np.array(v)**2).sum() ))


def _norm2(m):
	''' the induced 2 norm of matrix m (aka the spectral norm). This is the maximum singular value '''
	u,s,v = la.svd(m)
	return s.max()

def minAngle(vecs1, vecs2):
	'''
	return the minimal angle between the subspaces of R^n defined by the 	
	sets of projection vectors vecs1 and vecs2. These are (n,x) arrays
	with the vectors presented as columns. Vecs1 and vecs2 may have 
	different numbers of columns.

	Angle is returned in degrees.
	'''
	m = np.dot(P(vecs1), P(vecs2))
	return (180/np.pi)*np.arccos(_5sf(_norm2(m)))

def projM(v1):
	z = np.ones((v1.shape[0], 1))
	z = z/_norm(z)
	zp1 = np.dot(P(v1), z)
	return _norm(zp1)

def projSS(v1, v2):
	'''
	projector magintude assesment of subspace similarity
	'''
	p1 = P(v1)
	p2 = P(v2)
	v1v = v1.sum(1)
	v1v = v1v/ _norm(v1v)
	v2v = v2.sum(1)
	v2v = v2v/ _norm(v2v)
	return np.mean( (_norm(np.dot(p1, v2v)), _norm(np.dot(p2, v1v))) )

def jointSpan(v1, v2):
	'''
	returns an orthogonal basis of the combined space v1 union v2
	'''
	return sla.orth(np.column_stack([v1, v2]))

def sqrtPSDm(m):
	'''
	Uses SVD to calculate a matrix square root.
	input is the matrix, return is another matrix of the same shape.
	'''
	[u,s,v] = la.svd(m)
	s = np.mat(s*np.eye(s.shape[0]))
	u = np.mat(u)
	return u*np.sqrt(s)*u.transpose()

def gMod(ens1, ens2=None, whiten=True):
	'''
	For consistancy with the older convention, ens1 is conditioned, and 
	ens2 is unconditioned. If ens2 is white already, it may be ommited.
	'''
	m1 =np.mat( ens1.mean(1)).transpose()
	c1 =np.mat( np.cov(ens1, rowvar=1))
	if ens2 == None:
		return (m1, c1, np.mat(np.eye(m1.shape[0])), None)
	m2 =np.mat( ens2.mean(1)).transpose()
	c2 =np.mat( np.cov(ens2, rowvar=1))
	if not whiten:
		return (m1, c1, m2, c2)
	else:
		whiten = sqrtPSDm(la.inv(c2))
		mu = whiten*(m1-m2)
		co = whiten*c1*whiten
		return (mu, co, whiten, None)
	

