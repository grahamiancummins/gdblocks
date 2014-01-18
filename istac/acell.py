#!/usr/bin/env python -3
# encoding: utf-8

#Created by gic on 
#Wed Nov 17 12:09:39 CST 2010

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
from ll import compare as iscmp
import gicdat.search as gs
from gicdat.base import Transform
import gicdat.cext.slidingmatch as sm
from gwn import getstim
from gicdat.util import hist
import rand

'''
This module defines a collection of "artificial cells", which are simple mathematical models that represent some coding behavior

All artificial cells in this module share the same API. They take one input, s,
a timeseries <array[N]>, and return an event sequence <array[M] of int>.

There are also a number of helper functions in addition to the artificial
cells. Primary access to the artificial cells is through the module global list
CELLS, which contains all of them

'''


def peakdet(g, thresh, jitter=0, cid=0):
    '''
    Find peaks in the timeseries g <array[N]> and return an array of events
    <array[M] of int>, containing the sample IDs where the peaks occur

    Thresh determines how many peaks to detect. Peaks are choosen if they are
    higher than a threshold, which is set so that about the fraction thresh of
    all points in g are larger than this value.

    After detecting events, if jitter is non-zero, the events are perturbed by
    a random value drawn from a gaussian with std dev = jitter (and mean zero).
    This value is rounded to int.

    random jitter comes from the rand.evtjitter function, which ensures frozen
    random state. Different cids will get different jitter, but the same cid
    will always get the same jitter (in the sense of the same sequence of jitter
    random values. Changes in the jitter deviation, and in the number of events
    will still apply)

    '''
    thresh = (1 - thresh) * g.size
    ma = g.max()
    mi = g.min()
    bw = (ma - mi) / 10000.0
    h = hist(g, bw, mi)
    bid = np.nonzero(np.cumsum(h) > thresh)[0][0]
    thresh = mi + bid * bw
    hits = np.nonzero(g > thresh)[0]
    evts = []
    best = hits[0]
    for i in range(1, len(hits)):
        if hits[i] - 1 > hits[i - 1]:
            evts.append(best)
            best = hits[i]
        elif g[hits[i]] > g[best]:
            best = hits[i]
    evts.append(best)
    evts = np.array(evts)
    if jitter:
        evts = evts + rand.evtjitter(jitter, evts.shape, cid)
    return evts


def sgauss(stim, mean, icov, offsets):
    resp = np.zeros_like(stim)
    offsets = offsets.astype(np.int32)
    for i in range(offsets.max(), stim.size):
        v = stim[i - offsets] - mean
        resp[i] = np.dot(v, np.dot(icov, v))
    return resp


def gmatch(stim, offsets, mean, cov):
    '''
    A function to calculate the sliding-window log likelyhood of a gaussian
    model Stim a timeseries <array[N]>.  offsets and mean are <array[M]> M, and
    cov <array[M,M]>.  mean and cov specify an M-dimensional Gaussian model.
    offsets specifies which stimulus samples are used in the model. The output
    is <array[N]>, with output[i] equal to the activation of the
    gaussian model (mean, cov) by the vector stim[i-offsets]

    Activation is represented by only the exponential argument of the Gaussian
    (that is -.5 * (v-mean) * inv(cov) * (v-mean)') and leaves out the
    exponentiation, and normalization, so the result is not a likelyhood, but a
    value proportional to the log likelyhood.

    '''
    offsets = offsets.astype(np.int32)
    mean = mean.astype(np.float64)
    cov = cov.astype(np.float64)
    stim = stim.astype(np.float64)
    icov = np.linalg.inv(cov)
    r = sm.sgauss(stim, mean, icov, offsets)
    #r = np.exp(-.5*r)
    r = -.5 * r
    return r


M1 = ((1), (1), (.05))

M2 = ((5, 3, 1), (.2, -.5, 1), (.3, -1, 1, .2, -1, .1))


def gmatchCell(s, off, mu, sigma, cid=0, ejit=1.0, ethresh=.01):
    '''
    generic gaussian model using gmatch followed by peakdet(g, .01, 1).
    Off, mu, and sigma are direct parameters for gmatch. Sigma specifies the
    covariance in triu_incices order (that is, the c[:,0] followed by c[1,1:],
    followed by c[2,2:], etc).

    cid is a flag that peakdet uses to ensure "frozen" random state

    '''
    mu = np.array(mu)
    off = np.array(off)
    nd = off.shape[0]
    sig = convertSigma(sigma)
    g = gmatch(s, off, mu, sig)
    evts = peakdet(g, ethresh, ejit, cid=0)
    return evts


def convertSigma(s):
    if type(s) in [list, tuple]:
        s = np.array(s)
    if len(s.shape) == 2 and s.shape[0] == s.shape[1]:
        return s[np.triu_indices(s.shape[0])]
    else:
        nd = s.size
        nd = (np.sqrt(1 + 8 * nd) - 1) / 2
        nd = int(nd)
        sig = np.zeros((nd, nd))
        sig[np.triu_indices(nd)] = s
        for i in range(nd - 1):
            for j in range(i + 1, nd):
                sig[j, i] = sig[i, j]
        return sig


def uts(x):
    # (n**2+n)/2 = x
    # n**2 +n -2x = 0
    n = (-1 + np.sqrt(1 + 8 * x) ) / 2
    return n


class ACell(Transform):
#FIXME: Testing isn't working yet
#	sig = {#'stimulus':timeseries_t,
#			'off':gs.TupleOf(gs.WildCard('N'), gs.REAL),
#			'mu':gs.TupleOf(gs.WildCard('N'), gs.REAL),
#			'cid':gs.INT,
#			'egit':gs.REAL,
#			'ethresh':gs.REAL,
#			'sigma':gs.TupleOf(gs.WCFunction(gs.WildCard('N'), uts), gs.REAL),
#			'outpath':gs.STR}

    defaults = {'off': (5, 3, 1), 'mu': (.2, -.5, 1), 'sigma': (.3, -1, 1, .2, -1, .1), 'outpath': 'events', 'cid': 0,
                'stimpath': '->stim5_150'}


    def run(self, pars, out, messages):
        #stimulus, off, mu, sigma, outpath, cid):
        evts = self.callWith(gmatchCell, {'s': pars['stimpath._']}, pars)
        out[pars['outpath']] = {'tag': 'events', 'samplerate': pars['stimpath.samplerate'], '_': evts}
        messages.append('%i events generated' % evts.size)


acell = ACell()


class Compare(Transform):
#	sig = {'length':gs.INT,
#			 'lead':gs.INT,
#			 'compress':gs.Or('sig', 'istac', 'pca', 'no'),
#			 'clevel':gs.REAL,
#			 'testprop':gs.REAL,
#			 'bootstrap':gs.INT,
#			 #'evts1':NodePath(events_t),
#			 #'evts2':NodePath(events_t),
#			 #'stim':NodePath(timeseries_t),
#			 'outpath':gs.STR
#			 }
    defaults = {'length': 60, 'lead': 40, 'compress': 'no', 'clevel': 0.0, 'testprop': .2, 'bootstrap': 10}

    def run(self, doc, pars, out, messages):
        paths = dict([(n, doc[pars[n]]) for n in ['evts1', 'evts2', 'stim']])
        ll = self.callWith(iscmp, paths, pars)
        c12 = (ll[0, 0, 0] - ll[0, 1, 0]) / (ll[0, 0, 1] + ll[0, 1, 1])
        c21 = (ll[1, 1, 0] - ll[1, 0, 0]) / (ll[1, 0, 1] + ll[1, 1, 1])
        out[pars['outpath']] = min(c12, c21)
        messages.append('ll values %s' % (str(ll[..., 0]),))


compare = Compare()

if __name__ == "__main__":
    import gwn

    s = gwn.getstim()
    s['model.off'] = (9, 5, 1)
    s['model.mu'] = (.2, -.5, 1)
    s['model.sigma'] = (.3, -1, 1, .2, -1, .1)
    s['model.cid'] = 0
    s['model.outpath'] = 'events'
    s['model.stimpath'] = 'stim5_150'
    #print(ac.test(s, 'model'))
    d, r = acell(s, 'model')
    for st in r:
        print(st)
    s['model2.off'] = (5, 3, 1)
    s['model2.mu'] = (.2, -.5, 1)
    s['model2.sigma'] = (.3, -1, 1, .2, -1, .1)
    s['model2.cid'] = 0
    s['model2.outpath'] = 'events2'
    s['model2.stimpath'] = 'stim5_150'
    d = d.fuse(acell(s, 'model2')[0])
    d['stim'] = s['stim5_150']
    cmo = {'length': 60, 'lead': 40, 'istac': 0, 'testprop': .2, 'bootstrap': 10, 'evts1': 'events', 'evts2': 'events2',
           'stim': 'stim', 'outpath': 'fitness'}
    print(d.keys())
    r = compare(d, cmo)
    print(r[0]['fitness'], r[1])



