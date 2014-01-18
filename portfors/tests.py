#!/usr/bin/env python -3
# encoding: utf-8

#Created by Graham Cummins on May 25, 2011

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
import gicdat.doc as gd
import timingInfo as ti
import numpy as np
from gicdat.control import report
import numpy.random as npr
from gicdat.enc import flat
from timingInfo import window
#All times are assumed to be integer offsets in microseconds


def hpois3(r, t):
    n = npr.poisson(r)
    return np.unique(npr.randint(0, t, n))


def hpois2(r, t):
    return np.nonzero(npr.uniform(0, 1, t) < r / t)[0]


def hpois(r, t):
    ei = t / r
    resp = []
    li = 0
    while 1:
        i = npr.exponential(ei)
        li += i
        if li > t:
            break
        resp.append(li)
    return np.array(resp)


def sigmoid(mp, slp):
    rtp = int(np.ceil(mp + 5.0 / slp))
    x = np.arange(1, rtp)
    return 1.0 / (1.0 + np.exp((-1 * (mp - x)) * slp))


class AC(object):
    '''
    artificial responder class
    '''
    name = 'hpois'

    def __init__(self, rlen=100000, rate=.1, prec=0, nstims=16, **kwargs):
        '''
        rlen is the length of a response in 1 usec bins. rate is an average response
        rate in Hz,

        prec and amp are parameters intended for use by subclasse, governing the
        temporal precision and magnitude of stimulus modulation. (prec should be a
        chraracteristic scale in sample points)

        nstims is how many (integer) stimulus IDs there are

        Additional keyword arguments will be bound to attributes with the same names.

        calls self.calc at the end
        '''
        self.rlen = int(rlen)
        self.rate = float(rate)
        self.prec = float(prec)
        self.hist = []
        self.nstims = nstims
        self.rt = self.rlen / 1.0e6
        self.exp = self.rate * self.rt
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        self.calc()

    def calc(self):
        '''
        precalculate values used by r. This class calculates self.exp, the expected
        number of events.

        '''


    def __call__(self, sid=0):
        '''
        return a response event train. sid is the integer ID of the "stimulus"

        calls self.r(sid), which subclasses should override, and then returns the last element
        of self.hist (which r should set)
        '''
        self.r(sid)
        r = self.hist[-1]
        r = r[np.logical_and(r >= 0, r <= self.rlen)]
        return tuple(r.astype(np.int64))


    def r(self, sid):
        '''
        generate a response to stimulus sid. Store it as the last element of the list
        self.hist. This base class uses a uniform poisson proc with self.rate, and
        always sets self.hist to have a single value (this response).

        '''
        self.hist = [hpois(self.exp, self.rlen)]


class RC(AC):
    name = 'rate'

    def r(self, sid):
        n = 2 * self.exp * sid / self.nstims
        if n <= 0:
            self.hist = [np.zeros(0)]
        else:
            if hasattr(self, 'perfect'):
                self.hist = [np.unique(npr.randint(0, self.rlen, int(round(n))))]
            else:
                self.hist = [hpois(n, self.rlen)]


class PD(AC):
    '''
    uses an explicit, but sample independent probability distribution,
    for each stim
    '''
    name = 'pd'

    def calc(self):
        self.pd = self.exp * np.ones((self.rlen, self.nstims)) / self.rlen

    def r(self, sid):
        self.hist = [np.nonzero(npr.uniform(0, 1, self.rlen) < self.pd[:, sid])[0]]


class VC(AC):
    name = 'var'

    def calc(self):
        self.exp = int(round(self.exp))
        self.ms = np.linspace(0, self.rlen, self.exp + 2)[1:-1]

    def r(self, sid):
        sd = self.prec * sid
        r = []
        for i in range(self.exp):
            m = self.ms[i]
            if sd:
                e = int(npr.normal(m, sd, 1))
            else:
                e = int(m)
            r.append(e)
        self.hist = [np.array(r)]


class LC(AC):
    '''
    latency coder
    '''
    name = 'latency'

    def calc(self):
        self.exp = int(round(self.exp))
        if not hasattr(self, 'sd'):
            self.sd = self.prec / 2.0
        self.mlat = int(round(2 * self.sd))

    def r(self, sid):
        if self.exp == 0:
            self.hist = [np.zeros(0)]
            return
        l = self.mlat + sid * self.prec
        r = np.unique(npr.normal(l, self.sd, self.exp).astype(np.int64))
        self.hist = [np.unique(npr.normal(l, self.sd, self.exp).astype(np.int64))]


class JDC(AC):
    name = 'jitd'

    def calc(self):
        if not hasattr(self, 'sd'):
            self.sd = 1000
        if not hasattr(self, 'latsd'):
            self.latsd = self.prec / 2.0
        self.mfst = self.prec
        self.minisi = 2000

    def r(self, sid):
        fst = npr.normal(self.mfst, self.latsd)
        misi = (sid) * self.prec + self.minisi
        isi = npr.normal(misi, self.sd)
        self.hist = [np.array([fst, fst + isi])]


class BURST(AC):
    name = 'burst'

    def calc(self):
        if not hasattr(self, 'sd'):
            self.sd = 1000
        if not hasattr(self, 'latsd'):
            self.latsd = self.prec / 2.0
        if not hasattr(self, 'sprob'):
            self.sprob = .9
        if not hasattr(self, 'slope'):
            self.slope = 2
        if not hasattr(self, 'mpt'):
            self.mpt = self.exp - 1.0 / self.slope
        print(self.mpt, self.exp, 1.0 / self.slope)
        if not hasattr(self, 'mfst'):
            self.mfst = 3 * self.latsd
        self.minisi = 1000
        self.sprobs = sigmoid(self.mpt, self.slope)

    def r(self, sid):
        n = float(sid + 1) / self.nstims
        sprobs = self.sprobs * n
        eisi = self.prec * 2 * n
        st = max(npr.normal(self.mfst, self.latsd), 0)
        r = []
        z = npr.uniform(0, 1, len(sprobs))
        for i in range(len(z)):
            os = max(npr.normal(eisi, self.sd), 0)
            if z[i] <= sprobs[i]:
                r.append(st)
                os = max(os, self.minisi)
            st = st + os
        self.hist = [np.array(r)]


class AMP(AC):
    def r(self, sid):
        if hasattr(self, 'fixedn'):
            n = self.fixedn
        else:
            n = npr.poisson(self.exp)
        if n == 0:
            self.hist = [np.zeros(0)]
            return
        sep = self.rlen / (sid + 1)
        centers = sep * (np.arange(sid) + 1)
        o = npr.permutation(centers)
        c = o[np.arange(n) % centers.shape[0]]
        evts = []
        for cent in c:
            evts.append(npr.normal(cent, self.prec))
        self.hist = [np.array(evts)]


class DC(AC):
    name = 'dblts'

    def calc(self):
        self.exp = self.exp / 2.0
        if not hasattr(self, 'sd'):
            self.sd = 1000.0
        if not hasattr(self, 'minisi'):
            self.minisi = 800.0

    def r(self, sid):
        resp = hpois(self.exp, self.rlen)
        r2 = []
        misi = (sid) * self.prec + self.minisi
        for t in resp:
            r2.append(t)
            isi = npr.normal(misi, self.sd, 1)
            isi = max(isi, 1)
            r2.append(t + isi)
        self.hist = [np.array(r2)]


class TGC(AC):
    name = 'tgc'

    def calc(self):
        if not hasattr(self, 'sd'):
            self.sd = 1000.0
        self.m1 = self.rlen / 2.0 - self.prec / 2.0
        self.m2 = self.m1 + self.prec

    def r(self, sid):
        b = 2.0 * float(sid + 1) / (self.nstims + 2)
        first = b * self.exp / 2.0
        scnd = self.exp - first
        if hasattr(self, 'perfect'):
            first = int(round(first))
            scnd = int(round(scnd))
        else:
            first = npr.poisson(first)
            scnd = npr.poisson(scnd)
        r = np.concatenate([npr.normal(self.m1, self.sd, first), npr.normal(self.m2, self.sd, scnd)])
        self.hist = [r]


def draw_uniform(m, nstims, npres):
    evts = []
    stims = []
    for st in range(nstims):
        for _ in range(npres):
            stims.append(st)
            evts.append(m(st))
    return gd.Doc({'evts': evts, 'stims': stims})


def draw_single(m, sid, npres):
    evts = []
    stims = []
    for _ in range(npres):
        stims.append(sid)
        evts.append(m(sid))
    return gd.Doc({'evts': evts, 'stims': stims})


def cdoc(mods=(AC(100000, 20.), AC(100000, 100.)), nstims=16, npres=30):
    '''
    Return a cell conditions document using artificial data
    '''
    d = gd.Doc()
    for m in mods:
        cl = "cond%s" % (m.name,)
        d[cl] = draw_uniform(m, nstims, npres)
    return d


def p_cdoc():
    pars = (100000, 30, 15000)
    mods = [apply(c, pars) for c in [AC, RC, TGC]]
    d = cdoc(mods, 14, 30)
    sc = dict([('stim%i' % i, {'onset': 0, 'duration': 100}) for i in range(14)])
    d['stimclasses'] = sc
    return d


def tr(l=100000, r=1.0):
    a = AC(l, r)
    evts = [a(0) for i in range(1000)]
    report(len(flat(evts)) / 1000.0)


def tdoc(d, q=5000, mods=(AC, LC, VC, RC, DC, TGC), sp={}):
    td = gd.Doc()
    # (arate, nstims, nreps, l)
    arate = d[0]
    stims = []
    nstims = d[1]
    for si in range(d[1]):
        for _ in range(d[2]):
            stims.append(si)
    stims = np.array(stims)
    l = d[3]
    td['maxstimdur'] = l
    for M in mods:
        kw = sp.get(M.name, {})
        mi = M(l, arate, q, nstims, **kw)
        td['cond' + M.name] = {'stims': stims, 'evts': [mi(st) for st in stims]}
    td['docsource'] = {'cell': 'test', 'pars': tuple(d) + (q, )}
    return td


def addtests(d, l=200, match='cond1', q=10000, mods=(AC, TGC), sp={}):
    td = gd.Doc()
    iod = d[match]
    arate = float(len(flat(iod['evts']))) / len(iod['evts'])
    arate = arate / (l / 1e6)
    report(arate)
    stims = iod['stims']
    nstims = np.unique(stims).shape[0]
    l = int(l)
    td[match] = iod
    for k in d:
        if k.startswith('cond') and k != match:
            td[k] = d[k]
    for M in mods:
        kw = sp.get(M.name, {})
        mi = M(l, arate, q, nstims, **kw)
        td['cond' + M.name] = {'stims': stims, 'evts': [mi(st) for st in stims]}
    td['docsource'] = {'cell': d['docsource.cell'] + '+tests',
                       'pars': d['docsource.pars'] + (l, match, q)}
    return td


def addpois(d, conds, l=200):
    td = gd.Doc()
    for k in conds:
        td[k] = d[k]
        arate = float(len(flat(d[k]['evts']))) / len(d[k]['evts'])
        arate = arate / (l / 1e6)
        #report (arate)
        stims = d[k]['stims']
        mi = AC(l, arate)
        td[k + '_hp'] = {'stims': stims, 'evts': [mi(st) for st in stims]}
        td[k + '_hp.rate'] = arate
    td['docsource'] = d['docsource']
    return td
















    #