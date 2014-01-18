#!/usr/bin/env python 

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
from scipy.optimize import fmin_slsqp
import sys
import compress
from legacy import istacspace as istacspaceOld


LOGFILE = "istac.log"
#uncomment here to send logging to std out
LOGFILE = ""


def logdet(m):
    #return np.log(la.det(m))
    return np.trace(sla.logm(m))


def sqrtPSDm(m):
    '''
    Uses SVD to calculate a matrix square root.
    input is the matrix, return is another matrix of the same shape.
    '''
    [u, s, v] = la.svd(m)
    s = np.mat(s * np.eye(s.shape[0]))
    u = np.mat(u)
    return u * np.sqrt(s) * u.transpose()


def logcall(func, args, lfile='complog.txt'):
    of = open(lfile, 'a')
    so = sys.stdout
    sys.stdout = of
    try:
        r = apply(func, args)
    finally:
        sys.stdout = so
    return r


def istacspace(ece, uce, howmany, similar=False):
    mu, sig, whiten = compress.gMod(ece, uce)[:3]
    percentKL = False
    maxKL = compDklProj(mu, sig, None)
    if not howmany:
        howmany = mu.shape[0]
    elif howmany < 1:
        howmany = maxKL * howmany
        percentKL = True
    if LOGFILE:
        v, va = logcall(gausiSTAC, (mu, sig, howmany, percentKL, similar), LOGFILE)
    else:
        v, va = gausiSTAC(mu, sig, howmany, percentKL, similar)
    sp = {'vals': np.array(va), 'maxKL': maxKL}
    v = la.solve(whiten, v)
    sp['vecs'] = np.array(gsorth(v))
    return sp


def norm(v):
    '''
    v is a 1D array. The return is a float giving the 2-norm of the vector v
    '''
    return float(np.sqrt((np.array(v) ** 2).sum()))


def compDklProj(mu, A, vecs):
    '''
    mu and A specify a Gaussian model (these should be matrix instances with mu
    in row shape). The return is a float giving the KL divergence between this
    model and a sphericals models with mean zero and variance 1. Vecs specifies
    a subspace, and if it is not None, then the model will first be projted
    into this space, and then the KL calculated

    '''
    if vecs != None:
        mu = vecs.transpose() * mu
        A = vecs.transpose() * A * vecs
    d = .5 * (np.trace(A) - logdet(A) + np.dot(mu.transpose(), mu) - mu.size)
    return d[0, 0]


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
        k = np.array(vecs[:, j] - np.dot(B, (Binv * B.transpose() * vecs[:, j])))
        if norm(k) > etol:
            nv = nv + 1
            if vorth == None:
                vorth = k / norm(k)
            else:
                vorth = np.column_stack([vorth, k / norm(k)])
    return vorth


def negKLsubspace(k, mu, A, bv, vA, vAv, vecs):
    '''

    '''
    k = np.mat(k)
    if k.shape[0] == 1:
        k = k.transpose()
    if not vecs == None:
        k = k - vecs * (vecs.transpose() * k)
    k = k / norm(k)
    b1 = k.transpose() * mu
    v1 = k.transpose() * A * k
    if bv != None:
        b1 = np.row_stack([b1, bv])
        vAb = vA * k;
        v1 = np.row_stack([np.column_stack([v1, vAb.transpose()]),
                           np.column_stack([vAb, vAv])])
    L = logdet(v1) - np.trace(v1) - b1.transpose() * b1
    return L[0, 0]


def klsubspace(k, mu, A, bv, vA, vAv, vecs):
    l = negKLsubspace(k, mu, A, bv, vA, vAv, vecs)
    return -1 * l


def gs(v, B):
    '''Orthogonalizes v wrt B;	assumes that B is orthogonal'''
    v = v / norm(v)
    vnew = v - B * (B.transpose() * v)
    if norm(vnew) > 1e-10:
        vnew = vnew / norm(vnew)
    return vnew


def gsorth(a):
    m = a.shape[1]
    v = a[:, 0] / norm(a[:, 0])
    for j in range(1, a.shape[1]):
        v = np.column_stack([v, gs(a[:, j], v)])
    return v


def gausiSTAC(mu, A, howmany, percent, similar):
    '''

    '''
    vecs = None
    vals = []
    n = mu.shape[0]
    j = 0
    bv = None
    vA = None
    vAv = None
    [u, s, v] = la.svd(A)
    m = k0s = np.column_stack([u, mu / norm(mu)])
    # actually, you can use nearly anything here. The simplest is:
    #k0s = np.mat(np.eye(n))
    # and some backing up is avoided if the mean is also included
    if similar:
        klfunc = klsubspace
    else:
        klfunc = negKLsubspace
    while j < n:
        if percent:
            if len(vals) and vals[-1] >= howmany:
                break
        elif j >= howmany:
            break
        print 'iter', j
        BackingUP = 0
        if vecs != None and vecs.shape[1]:
            kstrt = orthogonalsubset(vecs, k0s)
        else:
            kstrt = k0s
        args = (mu, A, bv, vA, vAv, vecs)
        v0s = [klfunc(kstrt[:, ii], *args) for ii in range(kstrt.shape[1])]
        imin = np.argmin(v0s)
        k0 = kstrt[:, imin]

        def econs(x, mu, a, bv, vA, vAv, vecs):
            n = np.dot(x, x.transpose()) - 1
            r = [n]
            if vecs != None:
                q = np.dot(vecs.transpose(), x)
                q = list(np.array(q)[0, :])
                r.extend(q)
            return np.array(r)

        k = fmin_slsqp(klfunc, k0, f_eqcons=econs,
                       args=args, bounds=[(-1, 1)] * k0.shape[0])
        k = np.mat(k).transpose()
        if vecs != None:
            k = k - vecs * (vecs.transpose() * k)
            k = k / norm(k)
            vecs = np.column_stack([vecs, k])
        else:
            vecs = k
        vals.append(compDklProj(mu, A, vecs))
        if similar:
            valdiffs = [-vals[0]] + [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]
        else:
            valdiffs = [vals[0]] + [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        if BackingUP >= 3:
            BackingUP = 0
        elif (len(valdiffs) > 1 and valdiffs[j] > min(valdiffs[:-1])) and j < (n / 2) - 1:
            jj = np.nonzero(valdiffs[:-1] < valdiffs[-1])[0][0]
            k0s = np.column_stack([k, k0s])
            #print(vecs.shape, vals, valdiffs, jj)
            vecs = vecs[:, :jj]
            vals = vals[:jj]
            j = jj
            print(valdiffs)
            print('Going back to iter #%d (valdiff=%.4f)' % (j, valdiffs[-1]))
            BackingUP = 1
        elif j > 1:
            vv = vecs[:, [i for i in range(j) if not i == j - 2]]
            valtst = compDklProj(mu, A, vv)
            if valtst > vals[-2]:
                print('Wrong dim possibly stripped off [%.4f %.4f]; going back to prev dim' % (vals[-2], valtst))
                k0s = np.column_stack([k, k0s])
                vecs = vecs[:, :-2]
                vals = vals[:, :-2]
                j = j - 1
                BackingUP = BackingUP + 1
        if not BackingUP:
            j = j + 1
        bv = vecs.transpose() * mu
        vA = vecs.transpose() * A
        vAv = vA * vecs
    return (vecs, vals)

