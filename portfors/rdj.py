#!/usr/bin/env python -3
# encoding: utf-8

#Created by Graham Cummins on Jul 13, 2011

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

import tests as T
import gicdat.doc as gd
import dist
import clust
from timingUI import showcell
from gicdat.enc import flat
from gicdat.util import infdiag
import numpy as np
import matplotlib.pyplot as plt
from gicdat.gui.mpl import COLORS

'''
all uses of "q" in this module refer to the maximum distance that a spike will be translated. 

'''


def ucol(i, m):
    return plt.cm.spectral(float(i) / m)


def bitenc(a, bits, ar=None):
    nlev = 2 ** bits
    if ar == None:
        ar = [a.min(), a.max()]
    else:
        ar = list(ar)
        a = np.where(a < ar[0], ar[0], a)
        a = np.where(a > ar[1], ar[1], a)
    ar[1] = ar[1] - ar[0]
    a = (a - ar[0]) / ar[1]
    a = np.round(a * nlev).astype(np.int32)
    a = a.astype(np.float64) * ar[1] / nlev + ar[0]
    return a


def distance(lot, et, q, meth):
    if et == None:
        return dist.dist(lot, meth, q)
    else:
        return dist.dist_toset(et, lot, meth, q)


def vspaceav(st1, st2, q, wt1, wt2, dmeth):
    vr = dmeth.split('_')[1]
    st1 = dist.VREPS[vr]([st1], q)[0]
    st2 = dist.VREPS[vr]([st2], q)[0]
    av = (wt1 * st1 + wt2 * st2) / (wt1 + wt2)
    return dist.INVREPS[vr]([av], q)[0]


def vpd_trans(st1, st2, q):
    return dist.vdE(st1, st2, q)[1]


def dconds(d):
    return [k for k in d if k.startswith('cond')]


def td(nstim=3, nrep=100, rt=15.0, dur=100, prec=25.0, latsd=5000):
    dur = int(dur * 1000)
    prec = int(prec * 1000)
    sd = int(prec / 5.0)
    mods = [
        T.AC(dur, rt, prec, nstim),
        T.RC(dur, rt, prec, nstim),
        T.JDC(dur, rt, prec, nstim, sd=sd, latsd=latsd),
        T.BURST(dur, rt, prec, nstim, sd=sd, latsd=latsd),
    ]
    d = gd.Doc()
    for m in mods:
        d['cond%s' % (m.name,)] = T.draw_uniform(m, nstim, nrep)
    return d


def nburst(n=100, rt=15.0, dur=100, prec=15, sd=2000, latsd=5000, slp=2, mpt=3, sprob=.9):
    dur = int(dur * 1000)
    prec = int(prec * 1000)
    m = T.BURST(dur, rt, prec, 1, sd=sd, latsd=latsd, sprob=sprob, slp=slp, mpt=mpt)
    d = gd.Doc()
    d['condburst'] = T.draw_single(m, 0, n)
    rt = d['condburst.evts']
    rt = len(flat(rt)) / float(len(rt))
    rt = rt * 1e6 / dur
    print(rt)
    m = T.AC(dur, rt, prec, 3)
    d['condhpois'] = T.draw_single(m, 1, n)
    return d


def nscatter(reps=50, stims=[2, 3], pspike=.9, sd=2000, fixn=False):
    if fixn:
        m = T.AMP(100000, 10, sd, fixedn=pspike)
    else:
        m = T.AMP(100000, 10 * pspike, sd)
    d = gd.Doc()
    evts = []
    sts = []
    for s in stims:
        for _ in range(reps):
            sts.append(s)
            evts.append(m(s))
    d['condscatter'] = gd.Doc({'evts': evts, 'stims': sts})
    return d


def combinedraws(c1, c2):
    cc = gd.Doc()
    cc['stims'] = c1['stims']
    cc['evts'] = []
    for i in range(len(c1['evts'])):
        cc['evts'].append(sorted(c1['evts'][i] + c2['evts'][i]))
    return cc


def rburst(n=200, prate=10):
    dur = 100000
    d = gd.Doc()
    mp = T.AC(dur, prate, 1, 3)
    d['condpois'] = T.draw_single(mp, 0, n)
    m = T.BURST(dur, 1, 15000, 1, sd=1000, latsd=2000, mfst=10000, sprob=.8, slp=2, mpt=3)
    b = T.draw_single(m, 0, n)
    rt = 1e6 * len(flat(b['evts'])) / float(len(b['evts'])) / dur
    print('burst1 rate', rt)
    p = T.draw_single(mp, 0, n)
    rt = 1e6 * len(flat(p['evts'])) / float(len(p['evts'])) / dur
    print('pois rate', rt)
    d['condb1'] = combinedraws(b, p)
    m = T.BURST(dur, 1, 10000, 1, sd=2000, latsd=2000, mfst=15000, sprob=.9, slp=3, mpt=2)
    b = T.draw_single(m, 0, n)
    rt = 1e6 * len(flat(b['evts'])) / float(len(b['evts'])) / dur
    print('burst2 rate', rt)
    p = T.draw_single(mp, 0, n)
    d['condb2'] = combinedraws(b, p)
    return d


def distmat(d, dmeth, q, bits=8):
    for c in dconds(d):
        dm = distance(d[c]['evts'], None, q, dmeth)
        if bits:
            md = 2 * max([len(f) for f in d[c]['evts']])
            dm = bitenc(dm, bits, (0, md))
        d[c + '.dm'] = dm


def clustm(d, nc, mode='med', npass=1000):
    for c in dconds(d):
        dm = d[c]['dm']
        if mode == 'med':
            cd = clust.kmedoids(dm, nc, npass)[0]
        elif mode == 'tree':
            cd = clust.dtree(dm).cut(nc)
        cids = list(set(cd))
        cid = []
        for id in cd:
            cid.append(cids.index(id))
        d[c]['cid'] = cid


def howgrouped(stims, cids):
    sids = sorted(list(set(stims)))
    allcids = sorted(list(set(cids)))
    grps = {}
    for s in sids:
        grps[s] = dict([(k, 0) for k in allcids])
    for i, si in enumerate(stims):
        ci = cids[i]
        grps[si][ci] = grps[si][ci] + 1
    return grps


def testclust(d, q, dmeth='vdps', nc=None, bits=8, cmeth='med'):
    cnds = dconds(d)
    nstims = max([len(np.unique(d[c]['stims'])) for c in cnds])
    if not nc:
        nc = nstims
    distmat(d, dmeth, q, bits)
    clustm(d, nc, cmeth)
    n = len(cnds)
    f = plt.figure(1)
    plt.clf()
    for i, c in enumerate(cnds):
        print(c)
        cids = d[c]['cid']
        stims = d[c]['stims']
        sinc = howgrouped(stims, cids)
        cids = sorted(list(set(cids)))
        x = np.array(sorted(sinc))
        for k in x:
            print(k, sinc[k])
        sp = plt.subplot(1, n, i + 1)
        if i > 0:
            sp.yaxis.set_visible(False)
        sp.xaxis.set_visible(False)
        bot = np.zeros(x.shape[0])
        for j in cids:
            v = [sinc[s][j] for s in x]
            plt.bar(x, v, .8, color=ucol(j, len(cids)), bottom=bot)
            bot += np.array(v)
        plt.title(c)
    f.canvas.draw()


def getmedoids(part, dm):
    clusts = np.unique(part)
    out = []
    for c in clusts:
        inds = np.nonzero(part == c)[0]
        dms = dm[inds, :][:, inds]
        dmd = dms.sum(1)
        out.append(inds[np.argmin(dmd)])
    return out


def clustmmr_med(d, q, ncr=(1, 5), bits=8, dmeth='vdps', npass=1000, ponly=False):
    '''
    Check the response encoded in d for multiple modes. d should be a "condition
    document": e.g. it should have keys "evts" and "stims", and will typically be
    equivalent to doc[condX] for some X, if doc is a document returned by one of
    the test data generators in this module, such as nburst or nscatter. The
    test is performed by clustering the "evts" (this pays no attention to the
    values in "stims"), using the kmedoids approach applied to a distance matrix.
    The distance matrix is calculated with a precision of "bits", using a distance
    function specified by "dmeth", and using the precision parameter q

    "ncr" specifies a range of number of clusters to test (it is a tuple of
    (min, max), the centers tested will be range(ncr[0], ncr[1]+1)).

    "npass" is used internally by the clustering algorithm.

    The return value is a list of len(range(ncr[0], ncr[1]+1)) tuples. Each tuple
    contains (L, P, E), where L is a list of the cluster centers, P is an array
    giving the partition (the integer ID of the cluster center associated to each
    response in "evts"), and E is the expectation value of the distance from an
    element to its associated center.

    If ponly is True, the return value is only a list of partitions, rather than
    (L, P, E) tuples.

    '''
    dm = distance(d['evts'], None, q, dmeth)
    if bits:
        dm = bitenc(dm, bits)
    out = []
    for nc in range(ncr[0], ncr[1] + 1):
        part, err = clust.kmedoids(dm, nc, npass)[:2]
        cents = [d['evts'][i] for i in getmedoids(part, dm)]
        out.append((cents, part, err))
    if ponly:
        return [o[1] for o in out]
    else:
        return out


def clustmmr_mix(d, ncr=(1, 5)):
    '''
    Like clustmmr_med, but uses mixture-model based clusters. These clusters
    assume that all spikes occur independently and thus model a 1-dimensional
    distribution of spikes (in time)

    '''


def avst_vd(st1, st2, q, w1=1, w2=1):
    if len(st2) < len(st1):
        return avst_vd(st2, st1, q, w2, w1)
    trans = vpd_trans(st1, st2, q)
    if not trans:
        return st1
    tw = float(w1 + w2)
    ptrans = w1 / tw
    hastrans = [t[0] for t in trans]
    new = [z for z in st1 if not z in hastrans]
    for ts in trans:
        if ts[0] == -1:
            p = np.random.uniform(0, 1)
            if p > ptrans:
                new.append(ts[1])
        elif ts[1] == -1:
            p = np.random.uniform(0, 1)
            if p < ptrans:
                new.append(ts[0])
        else:
            new.append((ts[0] * w1 + ts[1] * w2) / tw)
    return sorted(new)


def addrow(m, r, d=np.inf):
    m = np.column_stack([m, r])
    r = np.concatenate([r, [d]])
    return np.row_stack([m, r])


def medoid(ets, q, dmeth):
    dm = distance(ets, None, q, dmeth)
    i = 0
    di = dm[0, :].sum()
    for j in range(1, dm.shape[0]):
        dj = dm[j, :].sum()
        if dj < di:
            i = j
            di = dj
    return ets[i]


def mdist(et, ets, q, dmeth):
    if et == None:
        et = medoid(ets, q, dmeth)
    return distance(ets, et, q, dmeth).sum()


def avst(st1, st2, q, wt1, wt2, dmeth):
    if dmeth in ['vd', 'vdps']:
        return avst_vd(st1, st2, q, wt1, wt2)
    elif "_" in dmeth:
        return vspaceav(st1, st2, q, wt1, wt2, dmeth)
    else:
        print('warning, using the fallthrough avst')
        p1 = wt1 / (wt1 + wt2)
        rn = np.random.uniform(0, 1)
        if rn > p1:
            return st2
        else:
            return st1


def pair_and_av(evts, dm, wts, q, dmeth):
    i, j = np.unravel_index(dm.argmin(), dm.shape)
    nst = avst(evts[i], evts[j], q, wts[i], wts[j], dmeth)
    nef = [x for x in range(len(evts)) if not x in [i, j]]
    evts = [evts[x] for x in nef]
    nd = distance(evts, nst, q, dmeth)
    evts = evts + [nst]
    dm = dm[nef, :][:, nef]
    dm = addrow(dm, nd, np.inf)
    nw = wts[i] + wts[j]
    wts = [wts[w] for w in nef] + [nw]
    return evts, dm, wts


def optq(evts, mode='med-vdps', reps=1, minq=1000, qstep=1000):
    q = minq - qstep
    expt = round(float(len(flat(evts))) / len(evts))
    nm = 0
    while nm < expt:
        print(nm, expt, q)
        q = q + qstep
        m = calcav(evts, mode, q, reps, True)
        nm = len(m)
    return q


def treeXformAv(evts, q, dmeth):
    dm = infdiag(distance(evts, None, q, dmeth))
    wts = np.ones(dm.shape[0])
    while len(evts) > 2:
        evts, dm, wts = pair_and_av(list(evts), dm, wts, q, dmeth)
    return avst(evts[0], evts[1], q, wts[0], wts[1], dmeth)


def randXformAv(evts, q, dmeth):
    wts = [1] * len(evts)
    while len(evts) > 2:
        i, j = np.random.randint(0, len(evts), 2)
        if i == j:
            continue
        ne = avst(evts[0], evts[1], q, wts[0], wts[1], dmeth)
        nw = wts[0] + wts[1]
        inds = [x for x in range(len(evts)) if not x in [i, j]]
        evts = [evts[i] for i in inds] + [ne]
        wts = [wts[i] for i in inds] + [nw]
    return avst(evts[0], evts[1], q, wts[0], wts[1], dmeth)


def nearest(targs, st):
    n = len(targs)
    ss = np.searchsorted(targs, st)
    v = []
    for i, vv in enumerate(ss):
        if vv >= n:
            v.append(n - 1)
        elif vv <= 0:
            v.append(0)
        elif (targs[vv] - st[i]) < (st[i] - targs[vv - 1]):
            v.append(vv)
        else:
            v.append(vv - 1)
    return v


def sfate(evts, q, tst=None, track=None, near=False):
    ef = flat(evts)
    tfs = []
    if tst:
        for e in evts:
            tfs.extend(vpd_trans(e, tst, q))
            ntrans = len(evts)
            holes = [0] * (len(tst) + 1)
    else:
        n = len(evts)
        ntrans = 0
        for i in range(n - 1):
            for j in range(i, n):
                ntrans += 1
                tfs.extend(vpd_trans(evts[i], evts[j], q))
        holes = False
    if track == None:
        if tst:
            track = tst
        else:
            track = sorted(set(ef))
    elif type(track) == int:
        ef = flat(evts)
        track = np.linspace(min(ef), max(ef), track)
    track = list(track)
    fates = [[] for _ in range(len(track))]
    for tf in tfs:
        if near:
            tf = list(tf)
            for i in range(len(tf)):
                if tf[i] != -1:
                    tf[i] = track[nearest(track, [tf[i]])[0]]
        if tf[0] == -1:
            try:
                id = track.index(tf[1])
                fates[id].append(np.inf)
            except ValueError:
                pass
        elif tf[1] == -1:
            try:
                id = track.index(tf[0])
                fates[id].append(-np.inf)
            except ValueError:
                if holes:
                    id = np.searchsorted(track, tf[0])
                    holes[id] += 1
        else:
            try:
                id = track.index(tf[0])
                fates[id].append(tf[1] - tf[0])
            except ValueError:
                pass
            try:
                id = track.index(tf[1])
                fates[id].append(tf[0] - tf[1])
            except ValueError:
                pass
    return (track, fates, ntrans, holes)


def ftsummary(ft, sgn=1, zfill=0, insdel=False):
    a = []
    t = []
    if not ft:
        return (0, 0, 0, 0)
    for f in ft:
        if np.abs(f) == np.inf:
            if insdel:
                a.append(insdel)
            else:
                a.append(np.sign(f))
            t.append(0)
        else:
            a.append(0)
            t.append(sgn * f)
    if zfill:
        a.extend([0] * (zfill - len(a)))
        t.extend([0] * (zfill - len(t)))
    a = np.array(a)
    t = np.array(t)
    return (t.mean(), t.std(), a.mean(), a.std())


def transprob_rep(av, evts, q):
    tfs = [vpd_trans(e, av, q) for e in evts]
    fates = []
    for _ in range(len(av) + 1):
        fates.append(np.zeros((len(tfs), 3)))
    for i, trans in enumerate(tfs):
        for tf in trans:
            if tf[0] == -1:
                id = av.index(tf[1])
                fates[id][i, 1] = 1
            elif tf[1] == -1:
                id = np.searchsorted(av, tf[0])
                fates[id][i, 2] += 1
            else:
                id = av.index(tf[1])
                fates[id][i, 0] = tf[1] - tf[0]
    res = []
    for j in range(len(av)):
        res.append((av[j], fates[j][:, 0].mean(), fates[j][:, 0].std(),
                    fates[j][:, 1].mean(), fates[j][:, 2].mean() ))
    res.append((-1, 0, 0, 0, fates[-1][:, 2].mean()))
    return res


def calcav(e, mode, q, nreps=1, best=False):
    avs = []
    if not mode:
        mode = 'med-vd'
    mode, dmeth = mode.split('-')
    for _ in range(nreps):
        if mode == 'med':
            av = medoid(e, q, dmeth)
        elif mode == 'r':
            av = randXformAv(e, q, dmeth)
        elif mode == 't':
            av = treeXformAv(e, q, dmeth)
        avs.append(av)
    if not best:
        return avs
    elif len(avs) == 1:
        return avs[0]
    else:
        i = np.argmin([mdist(r, e, q, dmeth) for r in avs])
        return avs[i]


def setav(d, mode='t-vdps', q=15000, nreps=20):
    for c in dconds(d):
        avs = calcav(d[c]['evts'], mode, q, nreps, False)
        dmeth = mode.split('-')[1]
        if nreps == 1:
            d[c]['avg'] = avs[0]
            d[c]['avgvar'] = transprob_rep(avs[0], d[c]['evts'], q)
            d[c]['avgs'] = None
        else:
            d[c]['avgs'] = avs
            i = np.argmin([mdist(r, d[c]['evts'], q, dmeth) for r in avs])
            d[c]['avg'] = avs[i]
            d[c]['avgvar'] = transprob_rep(avs[i], d[c]['evts'], q)


def cover(evts, bw):
    miv = min(evts)
    mav = max(evts)
    bins = np.arange(miv - bw, mav + 2 * bw, bw)
    return bins


def transprob_hist(evts, q, bw=2000):
    bins = cover(flat(evts), bw)
    n = len(evts)
    nt = (n ** 2 - n) / 2
    tfs = []
    for i in range(n - 1):
        for j in range(i, n):
            tfs.append(vpd_trans(evts[i], evts[j], q))
    fates = []
    for _ in range(len(bins)):
        fates.append(np.zeros((len(tfs), 2)))
    for i, trans in enumerate(tfs):
        for tf in trans:
            if tf[0] == -1:
                id = nearest(bins, [tf[1]])[0]
                fates[id][i, 1] = 1
            elif tf[1] == -1:
                id = nearest(bins, [tf[0]])[0]
                fates[id][i, 1] = 1
            else:
                id1 = nearest(bins, [tf[0]])[0]
                id2 = nearest(bins, [tf[1]])[0]
                dist = tf[1] - tf[0]
                fates[id1][i, 0] = dist
                fates[id2][i, 0] = -dist
    res = np.zeros((len(bins), 5))
    for i in range(len(bins)):
        res[i, 0] = bins[i]
        ns = float((fates[i][:, 0] != 0).sum() + (fates[i][:, 1] != 0).sum())
        res[i, 4] = ns / nt
        v = [x for x in fates[i][:, 0] if x != 0]
        if len(v) < ns:
            v = v + [0] * int(ns - len(v))
        v = np.array(v)
        res[i, 1] = v.mean()
        res[i, 2] = v.std()
        res[i, 3] = fates[i][:, 1].sum() / ns
    return res


def ll_et(evts, bw=2000):
    nt = len(evts)
    evts = flat(evts)
    spp = float(len(evts)) / nt
    bins = cover(evts, bw)
    h = np.histogram(evts, bins=bins, normed=True)
    h0 = h[0] * spp * bw
    zi = np.searchsorted(h[1], evts, 'right') - 1
    #	for i in range(len(h0)):
    #		print(h[0][i], (zi==i).sum())
    ll2 = np.log(h0[zi]).sum()
    return (ll2 / nt, (h0, h[1]))


def hdraw(h, st):
    return h[0][np.searchsorted(h[1], st, 'right') - 1]


def gausact(m, s, x, bw=1):
    n = 1.0 / (np.sqrt(2 * np.pi * s ** 2))
    e = -(x - m) ** 2 / (2 * s ** 2)
    return bw * n * np.exp(e)


def dj_rep(evts, q, amode='t-vdps', areps=20, bw=2000, nsteps=4, mltn=False):
    ll, sph = ll_et(evts, bw)
    while True:
        changes = 0
        start = sph[1][0]
        end = sph[1][-1]
        m = calcav(evts, amode, q, areps, True)
        tfp = transprob_rep(m, evts, q)
        delprobs = []
        intervals = [start] + list(m) + [end]
        for j in range(len(tfp)):
            iw = (intervals[j + 1] - intervals[j]) / float(bw)
            delprobs.append(tfp[j][4] / iw)
        nevts = []
        for e in evts:
            ttm = vpd_trans(e, m, q)
            ne = list(e)
            for t in ttm:
                if t[1] == -1:
                    sid = np.searchsorted(m, t[0])
                    tp = delprobs[sid]
                    sp = hdraw(sph, t[0])
                    if mltn:
                        if (1 - sp) * tp > sp * (1 - tp):
                            ne.remove(t[0])
                            changes = 1
                    elif (1 - sp) * tp > sp:
                        ne.remove(t[0])
                        changes = 1
                else:
                    sid = m.index(t[1])
                    if t[0] == -1:
                        tp = tfp[sid][3]
                        sp = hdraw(sph, t[1])
                        if mltn:
                            if sp * tp > (1 - sp) * (1 - tp):
                                ne.remove(t[0])
                                changes = 1
                        elif sp * tp > (1 - sp):
                            ne.append(t[1])
                            changes = 1
                    else:
                        sm = tfp[sid][1]
                        ssd = tfp[sid][2]
                        shifts = np.arange(-nsteps, nsteps + 1) * bw
                        shifts = np.array([s for s in shifts if s + t[0] > start and s + t[0] < end])
                        lc = [hdraw(sph, t[0] + s) * gausact(sm, ssd, s, bw) for s in shifts]
                        opts = shifts[np.argmax(lc)]
                        if opts != 0:
                            ne.remove(t[0])
                            ne.append(t[0] + opts)
                            changes = 1
            nevts.append(sorted(ne))
        if not changes:
            print('response set is unchanged')
            break
        nll, sph = ll_et(nevts, bw)
        if nll <= ll:
            print('totall ll decreased')
            break
        evts = nevts
        print(ll, '->', nll)
        ll = nll
    return evts


def dj_hist(evts, q, bw=2000, nsteps=4, insp=0):
    ll, sph = ll_et(evts, bw)
    ll = ll / float(len(flat(evts)))
    while True:
        start = sph[1][0]
        end = sph[1][-1]
        tp = transprob_hist(evts, q, bw)
        nevts = []
        for e in evts:
            ne = []
            for st in e:
                bid = nearest(tp[:, 0], [st])[0]
                sm, ssd, dp = tp[bid, 1:-1]
                #be careful, if the implementation of "cover" used by transprob_hist and
                #ll_et isn't exactly the same, this doesn't work
                sp = sph[0][bid]
                #check for deletion
                if (1 - sp) * dp * insp > sp:
                    continue
                shifts = np.arange(-nsteps, nsteps + 1) * bw
                shifts = np.array([s for s in shifts if s + st > start and s + st < end])
                lc = [hdraw(sph, st + s) * gausact(sm, ssd, s, bw) for s in shifts]
                opts = shifts[np.argmax(lc)]
                ne.append(st + opts)
            nevts.append(sorted(ne))
        nnspk = len(flat(nevts))
        if not nnspk:
            break
        nll, sph = ll_et(nevts, bw)
        nll = nll / nnspk
        if nll <= ll:
            break
        evts = nevts
        print(ll, '->', nll)
        ll = nll
    return evts


DJMETH = {'hist': dj_hist, 'rep': dj_rep}


def setdj(d, q, meth='hist', **kw):
    for c in dconds(d):
        e = d[c]['evts']
        ne = apply(DJMETH[meth], (e, q), kw)
        d[c]['dj'] = ne


def splitdjfuse(d, cond, part, q, meth, **kw):
    '''
    separately apply dejittering for groups in the partition "part".

    '''
    pids = np.unique(part)
    dj = []
    for pid in pids:
        inds = np.nonzero(part == pid)[0]
        evts = [d[cond]['evts'][i] for i in inds]
        dj.extend(apply(DJMETH[meth], (evts, q), kw))
    d[cond]['dj'] = dj


def part2cond(d, cond, part):
    nd = gd.Doc()
    st = d[cond]['stims']
    ev = d[cond]['evts']
    pids = np.unique(part)
    for p in pids:
        ind = np.nonzero(part == p)[0]
        nd['cond%i' % p] = {'stims': [st[i] for i in ind],
                            'evts': [ev[i] for i in ind]}
    return nd


def nhplt(evts, bins, color='r', bottom=0, scale=200):
    z = np.histogram(evts, bins=bins)
    y = scale * z[0] / z[0].sum()
    w = z[1][1] - z[1][0]
    plt.bar(z[1][:-1], y, width=w, color=color, bottom=bottom)
    return y.max() + 1


def showav(d, fig=1, yps=50, btop=100000, bw=2000):
    bins = np.arange(0, btop, bw)
    f = plt.figure(fig)
    plt.clf()
    cnds = dconds(d)
    n = len(cnds)
    yl = 0
    for i, c in enumerate(cnds):
        sp = plt.subplot(1, n, i + 1)
        sp.xaxis.set_visible(False)
        if i > 0:
            sp.yaxis.set_visible(False)
            sp.xaxis.set_visible(False)
        plt.title(c)
        l = len(d[c]['evts'])
        for j in range(l):
            evts = d[c]['evts'][j]
            if evts:
                x = np.array(evts)
                y = np.zeros_like(x) + j
                plt.plot(x, y, marker='.', color='r', linestyle='None')
        z = nhplt(flat(d[c]['evts']), bins, color='r', bottom=l + .2 * yps)
        yl = max(yl, l + 1.3 * yps)
        if d[c]['avgs']:
            z = nhplt(flat(d[c]['avgs']), bins, color='b', bottom=l + .1 * yps)
        if d[c]['dj']:
            for j in range(l):
                evts = d[c]['dj'][j]
                if evts:
                    x = np.array(evts)
                    y = np.zeros_like(x) + j
                    plt.plot(x, y, marker='.', color='g', linestyle='None')
            z = nhplt(flat(d[c]['dj']), bins, color='g', bottom=l + .1 * yps)
        if d[c]['avg'] != None:
            avl = int(l / 2.0)
            if len(d[c]['avg']) == 0:
                plt.axhspan(avl, avl + 1, color='b')
            else:
                x = np.array(d[c]['avg'])
                y = np.zeros_like(x) + avl
                plt.plot(x, y, marker='o', color='b', markersize=10.0, linestyle='None')
                if d[c]['avgvar'] != None:
                    av = d[c]['avgvar']
                    a = [0] + list(x) + [max(flat(d[c]['evts']))]
                    nx = [(a[i] + a[i - 1]) / 2.0 for i in range(1, len(a))]
                    ny = np.zeros_like(nx) + avl
                    ye = ([0] * len(av), [v[4] * yps for v in av])
                    plt.errorbar(nx, ny, yerr=ye, color='k', marker='.',
                                 markersize=4.0, linestyle='None', elinewidth=3)
                    av = av[:-1]
                    xmc = np.array([v[1] for v in av])
                    xe = [v[2] for v in av]
                    ymc = np.array([v[3] * yps for v in av])
                    plt.errorbar(x + xmc, y + ymc, xerr=xe, marker='s', color='b',
                                 markersize=6.0, linestyle='None', elinewidth=3)
    for i in range(len(cnds)):
        sp = plt.subplot(1, n, i + 1)
        plt.ylim([0, yl])
    f.canvas.draw()


def showpartitions(parts, evts, av="med-vdps", aq=15000, hists=True):
    '''
    Plot one or more partitions of a set of events. These are plotted as rasters,
    with events in each group of the partition shown together in a given color.
    Different partitions are shown in different subplots. "parts" should be a
    list of arrays, each specifing a partition. "evts" is a list of event
    sequences, av may be a False value, or a method string accepted by calcav. If
    it is specified, each group in the partition is also represented by an
    average calculated using this method, and "aq" as the precision parameter. If
    "hists" is true, the groups are represented with historgrams as well as
    rasters

    '''
    f = plt.figure(1)
    plt.clf()
    for pi, part in enumerate(parts):
        plt.subplot(1, len(parts), pi + 1)
        pids = np.unique(part)
        y = 0
        pcols = {}
        for i, pid in enumerate(pids):
            pcols[pid] = (COLORS[i], y)
            pevts = [evts[i] for i in range(len(evts)) if part[i] == pid]
            for j, e in enumerate(pevts):
                x = np.array(e)
                yc = np.zeros_like(x) + y
                plt.plot(x, yc, marker='.', color=pcols[pid][0], linestyle='None')
                y += 1
            if av:
                yav = float(pcols[pid][1] + y) / 2
                x = calcav(pevts, av, aq, 1, True)
                if len(x) == 0:
                    plt.axhspan(yav - .2, yav + .2, color=pcols[pid][0])
                else:
                    x = np.array(x)
                    yc = np.zeros_like(x) + yav
                    plt.plot(x, yc, marker='o', color=pcols[pid][0], markersize=10.0, linestyle='None')
        yz = 0
        if hists:
            for pid in pids:
                pevts = [evts[i] for i in range(len(evts)) if part[i] == pid]
                z = nhplt(flat(pevts), 50, color=pcols[pid][0], bottom=y + 1, scale=y)
                yz = max(z, yz)
        plt.ylim([-1, y + 2 + yz])
    f.canvas.draw()


def numnn(parts, evts, q=15000, t=.9):
    '''
    Calculate the number of non-null spike trains which are "isolated" in each
    partition in "parts", a list of partition centers, over "evts", using
    precision parameter "q". A spike train is "isolated" if its per spike victor
    distance to the nearest center is >= t (for the given q)

    '''
    num = []
    evts = [e for e in evts if len(e) > 0]
    for p in parts:
        dl = distance(evts, p[0], q, 'vdps') >= t
        n = 1
        while n < len(p):
            dl = np.logical_and(dl, distance(evts, p[n], q, 'vdps') >= 1)
            n += 1
        num.append(dl.sum())
    return num


def showpt(s):
    f = plt.figure(1)
    plt.clf()
    for i in range(len(s)):
        c = ucol(i, len(s))
        plt.plot([s[i][0]], [0], color=c, marker='o')
        plt.plot([s[i][0], s[i][0] + s[i][1]], [0, -s[i][3]], color=c)
        plt.plot([s[i][0], s[i][0]], [0, s[i][4]], color=c)
        plt.plot([s[i][0] + s[i][1] - s[i][2], s[i][0] + s[i][1] + s[i][2]], [-s[i][3], -s[i][3]], color=c)
    plt.ylim([-1, 1])
    f.canvas.draw()
	