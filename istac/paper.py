#!/usr/bin/env python
import gwn, acell, istac
import numpy as np
from ll import ece, eqrc, ucse, lltest, compare, DRMODES
import matplotlib.pyplot as plt
from gicdat.control import report


def evalSystem(m, ens=False, bl=True, cid=0):
    if not type(m[0]) in [tuple, list]:
        nd = int((-5 + np.sqrt(25 + 8 * len(m))) / 2)
        m = (m[:nd], m[nd:2 * nd], m[2 * nd:])
    if bl:
        stim = gwn.getstim('bl')
    else:
        stim = gwn.getstim('raw')
    evts = acell.gmatchCell(stim, m[0], m[1], m[2], cid)
    if not ens:
        return evts
    else:
        return ece(stim, evts)


def _llplt(ll):
    m = ll[..., 0].mean()
    plt.errorbar((1, 2, 3, 4), ll[..., 0].flatten() - m, ll[..., 1].flatten(), fmt='o', markersize=10)
    plt.plot((.5, 6.5), (0, 0), 'k-')
    plt.annotate("%.2f" % m, xy=(1, .1))
    y = ( ll[0, 0, 0] - ll[0, 1, 0], ll[1, 1, 0] - ll[1, 0, 0])
    err = ( ll[0, 0, 1] + ll[0, 1, 1], ll[1, 1, 1] + ll[1, 0, 1])
    plt.errorbar((5, 6), y, err, fmt='ro', markersize=10)
    plt.xticks((1, 2, 3, 4, 5, 6), ("1|1", "2|1", "1|2", "2|2", "1|1/2|1", "2|2/1|2"))
    plt.xlim(.5, 6.5)


def _prow(rc1, rc2, row, rows=3, normcov=True, ll=None, cpts=None):
    n = 2
    if rc2 != None:
        n += 1
        if ll != None:
            n += 1
    spos = n * row
    plt.subplot(rows, n, spos + 1)
    plt.plot(rc1.mean(1))
    c1 = np.cov(rc1)
    if cpts:
        for p in cpts[0]:
            x = 40 - p
            y = rc1.mean(1)[x]
            plt.annotate("%.3g" % y, xy=(x, y), xycoords='data',
                         xytext=(-10, -30), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->",
                                         connectionstyle="arc3,rad=.2")
            )
    plt.subplot(rows, n, spos + 2)
    i1 = plt.imshow(c1)
    if cpts:
        for p in cpts[0]:
            plt.axvline(x=40 - p, color='b')
    if n > 2:
        plt.subplot(rows, n, spos + 1)
        plt.plot(rc2.mean(1), 'r')
        if cpts:
            for p in cpts[1]:
                plt.axvline(x=40 - p, color=(.5, 0, 0))
        plt.subplot(rows, n, spos + 3)
        c2 = np.cov(rc2)
        i2 = plt.imshow(c2)
        if cpts:
            for p in cpts[1]:
                plt.axvline(x=40 - p, color='r')
        if not normcov:
            plt.colorbar(i2)
        else:
            cm = min(c1.min(), c2.min())
            cma = max(c1.max(), c2.max())
            i1.set_clim(cm, cma)
            i2.set_clim(cm, cma)
        if n > 3:
            plt.subplot(rows, n, spos + 4)
            _llplt(ll)


def modelFig(m1, m2=None, normcov=True):
    '''

    '''
    cpts = [m1[0], None]
    rcse = evalSystem(m1, True, cid=0)
    if m2:
        rcse2 = evalSystem(m2, True, cid=1)
        cpts[1] = m2[0]
    else:
        rcse2 = None
    plt.figure(1)
    plt.clf()
    _prow(rcse, rcse2, 0, 1, normcov, cpts=cpts)
    f = plt.figure(1)
    f.subplots_adjust(left=.04, right=.99, bottom=.1, top=.98, wspace=.05)
    f.canvas.draw()


def stacnbackFig(m1, stac=.8):
    stim = gwn.getstim('bl')
    uc = ucse(stim)
    rc = evalSystem(m1, True, cid=0)
    iss = istac.istacspace(rc, uc, stac)['vecs']
    plt.figure(3)
    plt.clf()
    _prow(rc, None, 0)
    rc1 = np.dot(iss.transpose(), rc)
    _prow(rc1, None, 1)
    rc2 = np.dot(iss, rc1)
    _prow(rc2, None, 2)
    f = plt.figure(3)
    f.subplots_adjust(left=.04, right=.99, bottom=.05, top=.99, wspace=.05, hspace=.05)
    f.canvas.draw()


def bigFig(m1, m2, compress='istac', clevel=.85):
    stim = gwn.getstim('bl')
    uc = ucse(stim)
    rcse = evalSystem(m1, True, cid=0)
    rcse2 = evalSystem(m2, True, cid=1)
    rc = eqrc((rcse, rcse2))
    iss = DRMODES[compress](rc, uc, clevel)
    report('Using %i components' % iss.shape[0])
    plt.figure(3)
    plt.clf()
    ll = lltest(rcse, rcse2)
    _prow(rcse, rcse2, 0, 3, True, ll)
    rc1 = np.dot(iss, rcse)
    rc2 = np.dot(iss, rcse2)
    ll = lltest(rc1, rc2)
    _prow(rc1, rc2, 1, 3, True, ll)
    rcb1 = np.dot(iss.transpose(), rc1)
    rcb2 = np.dot(iss.transpose(), rc2)
    #ll = lltest(rcb1, rcb2)
    _prow(rcb1, rcb2, 2, 3, True, ll)
    f = plt.figure(3)
    f.subplots_adjust(left=.04, right=.99, bottom=.05, top=.99, wspace=.05, hspace=.05)
    f.canvas.draw()


def llFig(m1, m2, compress='istac', clevel=.85):
    if istac:
        n = 2
    else:
        n = 1
    stim = gwn.getstim('bl')
    evts1 = evalSystem(m1, cid=0)
    evts2 = evalSystem(m2, cid=1)
    ll = compare(stim, evts1, evts2, compress='no')
    f = plt.figure(2)
    plt.clf()
    plt.subplot(1, n, 1)
    _llplt(ll)
    if istac:
        ll = compare(stim, evts1, evts2, compress=compress, clevel=clevel, report=report)
        plt.subplot(1, n, 2)
        _llplt(ll)
    f.subplots_adjust(left=.05, right=.98, bottom=.1, top=.98, wspace=.05)
    f.canvas.draw()


