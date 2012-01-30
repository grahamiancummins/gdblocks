#!/usr/bin/env python 

import numpy as np
import numpy.linalg as la
from scipy.optimize import fmin_slsqp
import sys
import compress

LOGFILE = "istac.log"
#uncomment here to send logging to std out
#LOGFILE = ""

def sqrtPSDm(m):
	'''
	Uses SVD to calculate a matrix square root.
	input is the matrix, return is another matrix of the same shape.
	'''
	[u,s,v] = la.svd(m)
	s = np.mat(s*np.eye(s.shape[0]))
	u = np.mat(u)
	return u*np.sqrt(s)*u.transpose()

def istacspace(ens1, ens2, howmany):
	'''
	Calculate an iSTAC subspace between two ensembles, ens1, and ens2. This is
	a subspace of the space spanned by the mean and covariance of the
	ensembles, and that maintains the largest fraction of the KL divergence
	between the ensembles that is possible given the size of the subspace. 

	ens1 and ens2 are the ensembles, represented as 2D arrays with columns
	being elements of the ensemble (and rows being individual samples within an
	element).

	howmany specifies the size of the subspace. If it is a positive integer, it
	as a number of dimensions. If it is a float greater than 0 and less than 1,
	keep enough dimensions to retain this fraction of the total KL divergence.

	The return value is a dictionary with keys:
	
	vecs - a 2D array representing the subspace. Each column is one of the unit
	vectors of the subspace.

	vals - a 1D array that specifies the KL divergence resulting from
	projection into a set of subspaces. vals[i] is the KL between the ensembles
	after projection into the i+1 dimensional subspace specified by vals[:,:i]
	
	maxKL - a float, specifying the KL between the raw ensembles.
	
	'''
	#ens are nreps x window length
	m1 =np.mat( ens1.mean(1)).transpose()
	m2 =np.mat( ens2.mean(1)).transpose()
	c1 =np.mat( np.cov(ens1, rowvar=1))
	c2 =np.mat( np.cov(ens2, rowvar=1))
	#print(c1.shape)
	of = open('compistaclog.txt', 'a')
	so = sys.stdout
	sys.stdout = of
	[v, va, maxKL] = compiSTAC(m1, c1, m2, c2, howmany)
	if howmany<1:
		print('using %i istac components\n' % v.shape[1])
	sys.stdout = so
	of.close()
	sp = {'vecs':v, 'vals':va, 'maxKL':maxKL}
	return sp		

def norm(v):
	'''
	v is a 1D array. The return is a float giving the 2-norm of the vector v
	'''
	return float(np.sqrt( (np.array(v)**2).sum() ))


def compDklgaussian(mu1,C1,mu2,C2):
	'''
	mu1 and C1 specify a multidimensional gaussian. mu2 and C2 specify another
	one (of the same dimension). Return is a float giving the KL divergence
	between the two Gaussians.

	Inputs have to be matrix instances, and the mu inputs should be in row
	vector shape (mu.shape[0] = 1, mu.shape[1] > 1)
	
	'''
	n = mu1.size
	b = mu2-mu1
	C2inv = la.inv(C2)
	C1sqrt =  np.mat(sqrtPSDm(C1))
	Term1 = C1sqrt*C2inv*C1sqrt
	Term2 = b.transpose()*C2inv*b
	det1 = la.det(C1)
	det2 = la.det(C2)
	tol = 1e8
	if (la.cond(C1)>tol) | (la.cond(C2)>tol):
		print('Determinants not non-zero. Ignoring determinants.')
		Term3 = 0
	else:
		Term3 = .5*np.log(det2/det1)
	d = .5*np.trace(Term1) + .5*Term2 - .5*n + Term3
	return d[0,0]

def compDklProj(mu, A, vecs):
	'''
	mu and A specify a Gaussian model (these should be matrix instances with mu
	in row shape). The return is a float giving the KL divergence between this
	model and a sphericals models with mean zero and variance 1. Vecs specifies
	a subspace, and if it is not None, then the model will first be projted
	into this space, and then the KL calculated

	'''
	if vecs != None:
		mu = vecs.transpose()*mu
		A = vecs.transpose()*A*vecs
	d = .5 * (np.trace(A) - np.log(la.det(A)) + np.dot(mu.transpose(),mu) - mu.size)
	return d[0,0]


def orthogonalsubset(B, vecs):
	'''
	vecs is a 2D array of column vectors. B is a subspace (also a 2D array of
	column vectors). Return is a specification of the part of vecs that is
	orthogonal to the subspace B

	'''
	etol = 1e-10
	Binv = np.mat(la.inv(np.dot(B.transpose(), B)))
	vorth = None
	nv = 0
	for j in range(vecs.shape[1]):
		k = np.array( vecs[:,j] - np.dot(B, (Binv*B.transpose()*vecs[:,j])))
		if norm(k) > etol:
			nv = nv+1
			if vorth == None:
				vorth = k/norm(k)
			else:
				vorth = np.column_stack([vorth, k/norm(k)])
	return vorth

def negKLsubspace(k, mu, A, bv, vA, vAv, vecs): 
	'''

	'''
	k = np.mat(k)
	if k.shape[0] == 1:
		k = k.transpose()
	if not vecs == None:
		k = k - vecs*(vecs.transpose()*k)
	k = k/norm(k)
	b1 = k.transpose()* mu
	v1 = k.transpose()*A*k
	if bv != None:
		b1 = np.row_stack([b1, bv])
		vAb = vA*k;
		v1 = np.row_stack( [np.column_stack([v1, vAb.transpose()]), 
							np.column_stack([vAb, vAv])] )
	L =  np.log(la.det(v1)) - np.trace(v1) - b1.transpose()*b1
	return L[0,0]

def klsubspace(k, mu, A, bv, vA, vAv, vecs):
	l = negKLsubspace(k, mu, A, bv, vA, vAv, vecs)
	return -1*l

def gs(v, B):
	'''Orthogonalizes v wrt B;	assumes that B is orthogonal'''
	v = v/norm(v)
	vnew = v-B*(B.transpose()*v)
	if norm(vnew) > 1e-10:
		vnew = vnew/norm(vnew)
	return vnew

def gsorth(a):
	m = a.shape[1]
	v = a[:,0]/ norm(a[:,0])
	for j in range(1,a.shape[1]):
		v = np.column_stack([v, gs(a[:,j], v)])
	return v
	

def compiSTAC(mu1, A1, mu0, A0, ndims):
	'''returns (vecs, vals, maxKL)'''
	vecs = None
	vals = []
	n = mu1.shape[0]
	UB = np.ones(n)
	LB = -1*np.ones(n)
	#opts = optimset('display', 'off', 'gradobj', 'off', 'largescale', 'off', ...
	# 'maxfunevals', 200000, 'maxiter', 50);
	A0whiten = sqrtPSDm(la.inv(A0));
	mu = A0whiten*(mu1-mu0)
	A = A0whiten*A1*A0whiten;
	[u,s,v] = la.svd(A)
	if ndims < 1 or ndims >= int(n/2):
		k0s = np.column_stack([u, mu/norm(mu)])
	else:
		iv = [i for i in range(n) if i < ndims or i>=n-ndims]
		k0s = np.column_stack([u[:,iv], mu/norm(mu)])
	bv = None
	vA = None
	vAv = None
	maxKL = compDklProj(mu, A, None )
	#print maxKL
	j = 0
	while j <n:
		if ndims >=1 and j >= ndims:
			break
		elif len(vals) and vals[-1]>= maxKL*ndims:
			break
		print 'iter',j
		BackingUP = 0
		if vecs!=None:
			kstrt = orthogonalsubset(vecs, k0s)
		else:
			kstrt = k0s
		args = (mu, A, bv, vA, vAv, vecs)
		v0s = [ negKLsubspace(kstrt[:,ii], *args) for ii in range(kstrt.shape[1])]
		imin = np.argmin(v0s)
		k0 = kstrt[:,imin]
		def econs(x, mu, a, bv, vA, vAv, vecs):
			n = np.dot(x, x.transpose()) - 1
			r = [n]
			if vecs!=None:
				q = np.dot(vecs.transpose(), x)
				q = list(np.array(q)[0,:])
				r. extend(q)
			return np.array(r)
		k = fmin_slsqp(negKLsubspace, k0, f_eqcons = econs,
			args = args, bounds = [(-1, 1)]*k0.shape[0])
		k = np.mat(k).transpose()
		if vecs !=None:	
			k = k-vecs*(vecs.transpose()*k)
			k = k/norm(k)
			vecs = np.column_stack([vecs, k])
		else:
			vecs = k
		vals.append( compDklProj(mu, A, vecs))
		valdiffs = [vals[0]] + [vals[i+1]-vals[i] for i in range(len(vals)-1)]
		if BackingUP >= 3:
			BackingUP = 0
		elif (len(valdiffs)>1 and valdiffs[j] > min(valdiffs[:-1])) and j<(n/2)-1:
			jj = np.nonzero(valdiffs[:-1] < valdiffs[-1])[0][0]
			k0s = np.column_stack([k, k0s])
			#print(vecs.shape, vals, valdiffs, jj)
			vecs = vecs[:,:jj]
			vals = vals[:jj]
			j = jj
			print('Going back to iter #%d (valdiff=%.4f)' % (j,valdiffs[-1]))
			BackingUP = 1
		elif j>1: 
			vv = vecs[:,[i for i in range(j) if not i==j-2]]
			valtst = compDklProj(mu, A, vv)
			if valtst > vals[-2]:
				print('Wrong dim possibly stripped off [%.4f %.4f]; going back to prev dim' % (vals[-2], valtst))
				k0s = np.column_stack([k, k0s])
				vecs = vecs[:,:-2]
				vals = vals[:,:-2]
				j = j-1
				BackingUP = BackingUP+1	
		if not BackingUP:
			j = j+1
		bv = vecs.transpose()*mu
		vA = vecs.transpose()*A
		vAv = vA*vecs
	vecs = la.solve(A0whiten, vecs)
	vecs = gsorth(vecs)
	return (np.array(vecs), np.array(vals), maxKL)

