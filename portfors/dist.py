#!/usr/bin/env python -3
# encoding: utf-8

#Created by gic on Thu Jan 20 11:44:48 CST 2011

# Copyright (C) 2011 Graham I Cummins
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
#from __future__ import print_function, unicode_literals
import numpy as np
import gicdat.cext.eventtrains as et
from gicdat.cext.distance import edist, cbdist #@UnresolvedImport
import gicdat.doc as gd
from gicdat.util import infdiag
import scipy.stats as st
import scipy.cluster.hierarchy as clust

squareform = clust.distance.squareform
from gicdat.enc import trange
from gicdat.util import CountTab

#Alternate distances

'''
all uses of "q" in this module refer to the maximum distance that a spike will
be translated.

Calls out to gicdat.cext.eventtrains ( et ) will use the more traditional
Victor/Purpura cost parameter, which is the cost of moving a spike by one
sample point. This cost is related to the max distance by et.q = 2.0/q

reimplimentation of the V/P algorithms is vdP, vdE internally use 2.0/q, but
should be passed the same max distance q as all other local functions in this
module

'''


def nullmax(isnull, dm):
    m = dm.max()
    dm[isnull, :] = m
    dm[:, isnull] = m
    return dm


def nctn(isnull, dm):
    m = dm.max()
    for i in range(isnull.shape[0]):
        if isnull[i]:
            for j in range(isnull.shape[0]):
                if not isnull[j]:
                    dm[i, j] = 2 * m
                    dm[j, i] = 2 * m
    return dm


def edist_set(e, s):
    e = np.array(e)
    s = np.array(s)
    diffs = s - e[np.newaxis, :]
    return np.sqrt((diffs ** 2).sum(1))


def cbdist_set(e, s):
    e = np.array(e)
    s = np.array(s)
    diffs = s - e[np.newaxis, :]
    return np.abs(diffs).sum(1)


def roundst(lot, q):
    nlot = []
    for t in lot:
        if not t:
            nlot.append(t)
        else:
            z = np.array(t, np.float64) / q
            z = np.round(z).astype(np.int64)
            nlot.append(tuple(z))
    return nlot


def binspace(lot, q=1):
    if q != 1:
        lot = roundst(lot, q)
    w = int(trange(lot)[1]) + 1
    r = np.zeros((len(lot), w))
    for i, t in enumerate(lot):
        for e in t:
            r[i, e] += 1
    return r


def binspace_inv(lot, q=1):
    evts = []
    for r in lot:
        et = np.nonzero(r)[0] * q
        evts.append(tuple(et))
    return evts


def _derez(v, q):
    n, rem = divmod(v, q)
    if rem >= q / 2.0:
        n += 1
    return q * n


def fixedlenst(lot, q=None):
    l = max([len(t) for t in lot])
    r = np.zeros((len(lot), l), np.float64) + np.inf
    for i, es in enumerate(lot):
        if q:
            es = [_derez(v, q) for v in es]
        r[i, :len(es)] = es
    return r


def fixedlenst_inv(lot, q):
    evts = []
    for r in lot:
        et = [v for v in r if not v == np.inf]
        evts.append(et)
    return evts


def isirep(evts, q=None):
    if q == None:
        start = 0
        l = max([len(t) for t in evts])
    else:
        start, l = q
    r = np.zeros((len(evts), l), np.float64)
    for i, es in enumerate(evts):
        last = start
        for j in range(l):
            try:
                v = float(es[j]) - last
                if q:
                    v = _derez(v, q)
                last = es[j]
            except IndexError:
                v = np.inf
            r[i, j] = v
    return r


def isirep_inv(lot, q=None):
    evts = []
    if q == None:
        start = 0
    elif type(q) in [list, tuple]:
        start = q[0]
    else:
        start = q
    for r in lot:
        if r[0] == np.inf:
            evts.append([])
            continue
        et = [start + r[0]]
        for isi in r[1:]:
            if isi == np.inf:
                break
            et.append(et[-1] + isi)
        evts.append(et)
    return evts


def vpdist(lot, q):
    return et.vpdistMatrix(lot, 2.0 / q)


def vpdist_set(lot1, lot2, q):
    return et.vpdistSet(lot1, lot2, 2.0 / q)


def vpdist_toset(st, lot, q):
    return et.vpdistSet([st], lot, 2.0 / q)[0, :]


def vpdist_ps(lot, q):
    q2 = 2.0 / q
    nl = len(lot)
    dm = np.zeros((nl, nl))
    for i in range(nl - 1):
        for j in range(i, nl):
            n = len(lot[i])
            m = len(lot[j])
            if not n or not m:
                d = 1
            else:
                d = et.vpdist(lot[i], lot[j], q2) / (n + m)
            dm[i, j] = d
            dm[j, i] = d
    return dm


def vpdist_toset_ps(st, lot, q):
    n = len(st)
    if n == 0:
        return np.ones(len(lot))
    d = np.ones(len(lot), np.float64)
    q2 = 2.0 / q
    for i, e in enumerate(lot):
        m = len(e)
        if m:
            d[i] = et.vpdist(st, e, q2) / (n + m)
    return d


def intervals(lot):
    '''
    return an array of interespike intervals (1D array of int) found in the
    response list lot (list of tuples of int). This icludes only intervals
    between two actually measured spikes, so responses with 0 or 1 spike
    contribute no intervals (which means it is possible for the response to be
    length 0).

    '''


def intlen(lot):
    '''
    Estimate a recording interval for the set of responses in lot (a list of
    tuples of integers). Return the estimate (a tuple of 2 floats, starttime,
    stoptime). The estimate is made by first estimating the expected value of the
    interspike interval, ISI, and then returning the minimum event time minus
    ISI/2 as the interval start, and the maximum plus ISI/2 as the stop.

    '''
    miv, mav = trange(lot)
    isi = intervals(lot).mean()
    return (miv - isi / 2.0, mav + isi / 2.0)


def _vpd_pw_pe(st1, st2, q, exp):
    if len(st1) == 0:
        if len(st2) == 0:
            return exp['nprob']


def vpdist_pe(lot, q):
    exp = {'nprob': float(len([x for x in lot if not x])) / len(lot)}

    q2 = 2.0 / q
    nl = len(lot)
    dm = np.zeros((nl, nl))
    for i in range(nl - 1):
        for j in range(i, nl):
            n = len(lot[i])
            m = len(lot[j])
            if not n or not m:
                d = 1
            else:
                d = et.vpdist(lot[i], lot[j], q2) / (n + m)
            dm[i, j] = d
            dm[j, i] = d
    return dm


def _inv(x):
    ze = x == 0
    if not ze.sum():
        return 1.0 / x
    xx = 1.0 / np.where(ze, 1, x)
    mm = xx.max() * 2
    return np.where(ze, mm, xx)


def vpdist_toset_pe(st, lot, q):
    pass


VDISTS = {'ed': edist, 'cb': cbdist}
VSETS = {'ed': edist_set, 'cb': cbdist_set}
NULLC = {'max': nullmax, 'nctn': nctn}
VREPS = {'raw': fixedlenst,
         'bin': binspace,
         'ist': lambda x, y: _inv(fixedlenst(x, y)),
         'isi': isirep,
         'irate': lambda x, y: _inv(1.0 / isirep(x, y))}
INVREPS = {'raw': fixedlenst_inv,
           'bin': binspace_inv,
           'ist': lambda x, y: fixedlenst_inv(_inv(x), y),
           'isi': isirep_inv,
           'irate': lambda x, y: isirep_inv(inv(x), y)}
DMATS = {'vd': vpdist, 'vdps': vpdist_ps, 'vdpe': vpdist_pe}
DTSETS = {'vd': vpdist_toset, 'vdps': vpdist_toset_ps, 'vdpe': vpdist_toset_pe}


def dist(lot, mode='vd', q=None, nulls=None):
    '''
    return the distance matrix of distances between event trains in lot (list of
    tuples of int) uses the distance function "mode" (a key into DMATS), with
    precision parameter q, and additional arguments "args". If "nulls" is not
    None, then the distance matrix is "null corrected" using the method named by
    "nulls" (which in a key into NULLC)

    q should be a characteristic discriminable length in microseconds.

    '''
    if "_" in mode:
        mode, vr = mode.split('_')
        dm = VREPS[vr](lot, q)
        dm = VDISTS[mode](dm)
    else:
        dm = apply(DMATS[mode], (lot, q))
    if nulls != None:
        isnull = np.array([len(l) == 0 for l in lot])
        dm = NULLC[nulls](isnull, dm)
    return dm


def classDM(dm, classes, nclasses=None):
    '''
    dm is a distance matrix (N,N-# of x), and classess is a N-[ of i where the
    integers form a partition of the rows of dm, which should almost always use
    sequential numbering from 0 to (number of classes -1). The return (ret) is a
    M-[ of [ of [ of x::M==max(classes)+1, where the list ret[a][b], b<=a, gives
    the set of all pairwise distances between members of class a and class b.
    When a==b, the 0 distances between a response and itself are omitted.
    '''
    classes = np.array(classes)
    if nclasses == None:
        nclasses = classes.max() + 1
    ret = []
    for i in range(nclasses):
        jl = []
        cl_i = np.nonzero(classes == i)[0]
        if cl_i.shape[0] == 0:
            ret.append([[] for j in range(i + 1)])
            continue
        for j in range(i):
            cl_j = np.nonzero(classes == j)[0]
            if cl_j.shape[0] == 0:
                jl.append([])
            else:
                jl.append(list(dm[cl_i, :][:, cl_j].ravel()))
        i_to_i = dm[cl_i, :][:, cl_i] - np.eye(cl_i.shape[0])
        jl.append([z for z in i_to_i.ravel() if z >= 0])
        ret.append(jl)
    return ret


def dist_toset(e, lot, mode='vd', q=None):
    if len(e) and hasattr(e[0], '__iter__'):
        raise ValueError('reverse lot and e arguments, or eventrains will segv')
    if "_" in mode:
        mode, vr = mode.split('_')
        lot = VREPS[vr](list(lot) + [e], q)
        e = lot[-1]
        lot = lot[:-1]
        dm = VSETS[mode](e, lot)
    else:
        dm = apply(DTSETS[mode], (e, lot, q))
    return dm


def confusion(stims, dm):
    '''
    stims is a tuple of stimulus conditions.

    dm is a distance matrix of responses (with the same number of rows as stims)

    Returns a confusion matrix between stimulus conditions
    '''
    di = infdiag(dm)
    stimind = np.array(stims)
    nstims = stimind.max() + 1
    conf = np.zeros((nstims, nstims))
    for i in range(di.shape[0]):
        inds = np.nonzero(di[i, :] == di[i, :].min())[0]
        st = stimind[inds]
        sst = stimind[i]
        n = 1.0 / inds.shape[0]
        for s in st:
            conf[sst, s] += n
            conf[s, sst] += n
    return conf

# Special distance calculations (e.g. transform trackers, etc)

def vdP(st1, st2, q):
    '''
    simple calculation of V/P distance between spike trains st1 and st2 with cost q
    The implementation in gicdat.cext.eventtrains.vpdist is much faster. This is present for
    references, testing, and documentation of the algorithm.

    st1 and st2 are sequences of event times. Q is the inverse V/P cost (this is the
    max number of sample points that it is possible to move a spike. The cost of moving a
    spike by one sample is 2.0/q)

    the return value is a float (the distance)
    '''
    #(52050, 72950, 76250, 139775)
    #(66475, 69725, 186750)
    #.0001 -> 4.3
    q = 2.0 / q
    if not st1:
        return len(st2)
    elif not st2:
        return len(st1)
    lasti = np.arange(len(st2) + 1).astype(np.float64)
    last = 1
    for i in range(1, len(st1) + 1):
        if i > 1:
            lasti[len(st2)] = last
        last = i
        for j in range(1, len(st2) + 1):
            choices = [
                lasti[j] + 1,
                last + 1,
                lasti[j - 1] + q * abs(st1[i - 1] - st2[j - 1])
            ]
            darg = np.argmin(choices)
            dist = choices[darg]
            print("(%i,%i) %i -> %.2f  (%.2f, %s)" % (i, j, darg, dist, last, str(lasti)))
            lasti[j - 1] = last
            last = dist
    return dist


def vdC(st1, st2, q):
    '''
    Should return the same thing as vpP, but much faster. Calls et.vpdist(st1, st2, 2.0/q)
    '''
    return et.vpdist(st1, st2, 2.0 / q)


def _inroi(st, roi):
    return np.array([s for s in st if s >= roi[0] and s <= roi[1]])


def vdSl(st1, st2, q, roi):
    '''
    Return a tuple (s1, s2) containing representations of input spike trains st1 and st2 that
    are appropriate for calculating a V/P distance on the interval roi
    st1 and st2 are sequences of spike times (or indexes), Q is the inverse V/P cost (this is the
    max number of sample points that it is possible to move a spike. The cost of moving a
    spike by one sample is 2.0/q)
    '''
    q = 2.0 / q
    t1 = _inroi(st1, roi)
    t2 = _inroi(st2, roi)
    if q == 0:
        return np.abs(len(t1) - len(t2))
    pad = 1.0 / q
    tr = (roi[0] - pad, roi[1] + pad)
    if len(t1) <= len(t2):
        t1 = _inroi(st1, tr)
    else:
        t2 = t1
        t1 = _inroi(st2, tr)
    nsp = len(t2) - len(t1)
    if nsp > 0:
        seq = np.arange(nsp).astype(t1.dtype)
        t1 = np.concatenate([tr[0] - seq[::-1], t1, tr[1] + seq])
    return (t1, t2)


def vdE(st1, st2, q):
    '''
    Calculates the V/P distance between spike trains st1 and st2 with inverse cost q
    Returns a tuple (d, ops) where d is the distance, and ops is a list of the operations
    used to render the spike trains equivalent. These operations are each 2-tuples of spike
    indexes. A -1 represents non-existance, so (400, -1) is "delete the spike at index 400,
    (-1, 400) is "insert a spike at index 400". Tuples not containing a -1 represent shifting,
    e.g. (400, 410) "move the spike found at index 400 (in st1) to index 410 (where it will
    correspond to a spike in st2)
    '''
    q = 2.0 / q
    if not st1:
        return (len(st2), zip([-1] * len(st2), st2))
    elif not st2:
        return (len(st1), zip(st1, [-1] * len(st1)))
    st1 = np.array(st1)
    st2 = np.array(st2)
    lasti = np.arange(len(st2) + 1).astype(np.float64)
    lasti = [(lasti[i], zip([-1] * i, st2[:i])) for i in range(len(lasti))]
    dist = (0, [])
    last = -1
    for i in range(1, len(st1) + 1):
        if i > 1:
            lasti[len(st2)] = last
        last = (i, zip(st1[:i], [-1] * i))
        for j in range(1, len(st2) + 1):
            if st1[i - 1] == st2[j - 1]:
                #dist doesn't change
                dist = lasti[j - 1]
            else:
                kill = lasti[j][0] + 1
                synth = last[0] + 1
                shift = lasti[j - 1][0] + q * abs(st1[i - 1] - st2[j - 1])
                if kill < min(synth, shift):
                    dist = (kill, lasti[j][1] + [(st1[i - 1], -1)])
                elif synth <= min(kill, shift):
                    dist = (synth, last[1] + [(-1, st2[j - 1])])
                else:
                    dist = (shift, lasti[j - 1][1] + [(st1[i - 1], st2[j - 1])])
            lasti[j - 1] = last
            last = dist
    return dist


def vdV(st1, st2, q, l='auto', fold=100):
    '''
    Return a V/P distance from spike train st1 to st2 with inverse cost q,
    represented as a temporal cost histogram. l and fold control the length and
    bin-width of this histogram
    '''
    if l == 'auto':
        l = max([max(st1), max(st2)]) + 1
    v = np.zeros(l)
    d = vdE(st1, st2, q)[1]
    for op in d:
        s = (min(op), max(op))
        if s[0] == -1:
            v[s[1]] += 1
        else:
            n = s[1] - s[0]
            v[s[0]:s[1]] += np.ones(n) * (2.0 / q)
    if fold:
        nw, lo = divmod(l, fold)
        if lo:
            v = v[:fold * nw]
        v = np.reshape(v, (-1, fold)).sum(1)
    return v


def wherediff(q, c1, c2=(), fold=100, sum=True):
    '''
    calculate the vdV histograms for each distance in class c1 and (optionally)
    between c1 and class c2, with inverse cost q, l auto, and the specified fold.
    '''
    ran = trange(c1 + c2)
    l = ran[1] + 1
    nw, lo = divmod(l, fold)
    if lo:
        l = (nw + 1) * fold
    dvs = []
    if c2:
        #cross class difference
        for s1 in c1:
            for s2 in c2:
                dvs.append(vdV(s1, s2, q, l, fold))
    else:
        for i in range(len(c1) - 1):
            for s2 in c1[i + 1:]:
                dvs.append(vdV(c1[i], s2, q, l, fold))
    if sum:
        dvs = np.array(dvs).sum(0)
    return dvs

