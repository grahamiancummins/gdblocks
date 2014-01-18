#!/usr/bin/env python -3
# encoding: utf-8

#Created by gic on 
#Mon Feb 28 10:32:55 CST 2011

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
from __future__ import print_function, unicode_literals
from acell import acell, convertSigma
from gicdat.control import report
from gicdat.gui.mpl import COLORS, LSTYLES
from ll import ece, ucse, eqrc, DRMODES
from twostim import compare2
import compress as com
import gicdat.doc as gd
import gwn
import re
import paper
import istac
import time
import os
import matplotlib.pyplot as plt
import numpy as np

STIMS_032211 = ((4, -5, 90), (4, 5, 90))
STIMS_CLOSE = ((10, -1, 90), (10, 1, 90))
STIMS_BAND_OL = ((5, 300), (100, 395))
#===============================================================================
# #Test systems are specified (off, mean, cov)
#===============================================================================
M200_BC = ( (5, 3, 1), (.2, -.5, 1), (.3, -1, 1, .2, -1, .1))
M200_RO = ( (7, 4, 1), (.2, -.5, 1), (.3, -0.1, .1, .2, -0.1, .1))
M200_FO = ( (7, 4, 1), (1, -1, 1), (.3, -0.1, .1, .2, -0.1, .1))
M200_2O = ( (13, 10, 7, 4, 1), (1, -1, 1, -1, 1), (.5, -.1, .1, -.1, .1, .4, -.1, .1, -.1, .3, -0.1, .1, .2, -0.1, .1))
M200_HO = ( (4, 1), (-1, 1), (.2, -0.1, .1))
M100_FO = ( (13, 7, 1), (1, -1, 1), (.3, -0.1, .1, .2, -0.1, .1))

FIGDIR = os.path.join(os.environ['HOME'], "Desktop")

usecase = '''
d = runSystems()
describe(d)
pltFirstStac(d)
pltSndStac(d)

'''


def savefig(label, fign, n=1):
    f = plt.figure(n)
    dir = os.path.join(FIGDIR, label)
    if not os.path.exists(dir):
        os.mkdir(dir)
    fn = os.path.join(dir, fign + '.png')
    plt.savefig(fn, format='png')
    f.canvas.draw()


def runSystems(stimuli=STIMS_BAND_OL, systems=(M200_FO,), stimlength=800, length=60, lead=40,
               label="A", date="", comments="", edet=(1.0, .01)):
    '''
    stimuli is a list of "band" parameter tuples. Systems is a list of
    (off, mean, cov) test system parameter tuples. "l" is a stimulus length.
    Constructs each stimulus in the stimulus list, and passes it to each
    system.
    '''
    if not date:
        date = time.strftime("%m%d%y")
    s = gd.Doc()
    s['label'] = 'iSTACe' + date + label
    s['comments'] = comments
    s['samplerate'] = 1000.0
    s['stimlength'] = stimlength
    s['enslength'] = length
    s['enslead'] = lead
    s['eventgenerator'] = 'peakdet'
    for i, b in enumerate(stimuli):
        sn = "s%i" % i
        s[sn + '._'] = gwn.getstim('bl', stimlength, b)
        s[sn + '.band'] = b
        un = "u%i" % i
        s[un + '._'] = ucse(s[sn + '._'], 10000, length)
    for i, p in enumerate(systems):
        mn = "m%i" % i
        off, mu, sig = p
        d = {'mu': mu, 'off': off, 'sigma': sig, 'cid': i, 'ejit': edet[0], 'ethresh': edet[1]}
        s.set(mn, d)
        for j in range(len(stimuli)):
            en = "e%i_%i" % (i, j)
            sn = "s%i" % j
            d['outpath'] = en
            d['stimpath'] = "->" + sn
            s = s.fuse(acell(s, d)[0])
            s[en + '.stim'] = {'_link': sn}
            s[en + '.sys'] = {'_link': mn}
            rn = "r%i_%i" % (i, j)
            s[rn + '._'] = ece(s[sn + '._'], s[en + '._'], length, lead)
            s[rn + '.stim'] = {'_link': sn}
            s[rn + '.sys'] = {'_link': mn}
    return s


def _keys(d, l):
    r = re.compile("%s(\d+)(_(\d+))?" % l)
    m = []
    for k in d.keys():
        rm = r.match(k)
        if rm:
            s = rm.group()
            d1, junk, d2 = rm.groups()
            m.append((s, d1, d2))
    return m


def _ntstr(t):
    fs = "%.2g," * len(t)
    fs = "(" + fs[:-1] + ")"
    s = fs % t
    return s


def describeS(d, fig=1):
    '''
    Create a figure describing the set of stimuli used in a document
    '''

    skeys = _keys(d, 's')
    plt.figure(fig)
    plt.clf()
    plt.subplot(211)
    sdesc = ''
    n = len(skeys)
    covs = []
    cmin = np.inf
    cmax = -np.inf
    plt.subplot(211)

    for k in skeys:
        s, i = k[:2]
        i = int(i)
        spec = gwn.spectrum(d[s]['_'])[:, 0]
        sd = "$S_%i=S(%i, %s)$" % (i, d['stimlength'], d[s + '.band'] )
        plt.plot(spec, label=sd)
        sdesc += ", " + sd
        ucse = d['u%i._' % i]
        c = np.cov(ucse)
        cmin = min(cmin, c.min())
        cmax = max(cmax, c.max())
        covs.append(('cov(U($S_%i$, %i, %i))' % (i, ucse.shape[1], ucse.shape[0]), c))
    plt.legend()
    plt.title("%s: Power Spectra of Stimuli" % d['label'])
    for i, c in enumerate(covs):
        plt.subplot(2, n, n + i + 1)
        plt.title(c[0])
        plt.imshow(c[1], vmin=cmin, vmax=cmax, interpolation='nearest')
        plt.xlim([0, c[1].shape[1]])
        plt.ylim([0, c[1].shape[0]])
    plt.figtext(0, .97, sdesc[2:])
    savefig(d['label'], 'Stimuli', fig)


def _pad(l, p=.1):
    r = [min(l), max(l)]
    p = p * (r[1] - r[0])
    return [r[0] - p, r[1] + p]


def describeM(d, fig=1):
    '''
    Create a figure describing the set of models used in a document
    '''
    leng = d['enslength']
    lead = d['enslead']
    models = []
    keys = _keys(d, 'm')
    nk = len(keys)
    plt.figure(fig)
    plt.clf()
    covs = []
    cmin = np.inf
    cmax = -np.inf
    plt.subplot(211)
    memin = np.inf
    memax = -np.inf
    for mod in keys:
        n, i = mod[:2]
        off = d[n + ".off"]
        mu = d[n + '.mu']
        memin = min(min(mu), memin)
        memax = max(max(mu), memax)
        sig = d[n + '.sigma']
        models.append("$M_%i = M(%s, %s, %s, %.3g, %.2g, %i)$" % (int(i), _ntstr(off),
                                                                  _ntstr(mu), _ntstr(sig), d[n + '.ethresh'],
                                                                  d[n + '.ejit'], d[n + '.cid']))
        off = lead - np.array(off)
        mean = np.zeros(leng)
        mean[off] = mu
        plt.axhline(0, color='black', lw=2)
        plt.vlines(np.arange(mean.shape[0]), 0,
                   mean, label="$M_%i$" % (int(i),),
                   color='k', linestyles='solid')
        clab = '$M_%i$ covariance' % (int(i),)
        sig = convertSigma(sig)
        cov = np.zeros((leng, leng))
        for i, ii in enumerate(off):
            for j, jj in enumerate(off):
                cov[ii, jj] = sig[i, j]
        covs.append((clab, cov))
        cmin = min(cmin, cov.min())
        cmax = max(cmax, cov.max())
    plt.ylim(_pad([memin, memax]))
    plt.title("%s: Model means, embedded in (%i, %i) sample window" % (d['label'], lead, leng))
    plt.legend()
    for i, c in enumerate(covs):
        plt.subplot(2, nk, nk + i + 1)
        plt.title(c[0])
        plt.imshow(c[1], vmin=cmin, vmax=cmax, interpolation='nearest')
        plt.xlim([0, c[1].shape[1]])
        plt.ylim([0, c[1].shape[0]])

    models = '\n'.join(models)
    plt.figtext(0, .03, models, backgroundcolor='white')
    savefig(d['label'], 'Models', fig)


def describeE(d, fig=1):
    '''
    Create a figure describing the conditional ensembles used in a document
    '''
    leng = d['enslength']
    lead = d['enslead']
    plt.figure(fig)
    plt.clf()
    covs = []
    rs = _keys(d, 'r')
    nk = len(rs)
    plt.subplot(211)
    plt.title('%s: Means of Conditional Ensembles (%i, %i) window' % (d['label'], lead, leng))
    mincv = np.inf
    maxcv = -np.inf
    models = {}
    for mod in _keys(d, 'm'):
        n, i = mod[:2]
        i = int(i)
        off = d[n + ".off"]
        mu = d[n + '.mu']
        off = lead - np.array(off)
        mean = np.zeros(leng)
        mean[off] = mu
        plt.plot(mean, label="Model $M_%i$" % i)
        models[i] = (min(off) - 1, max(off) + 1)
    for r in rs:
        n, i, j = r
        i = int(i)
        j = int(j)
        rcc = np.cov(d[n]['_'])
        mincv = min(mincv, rcc.min())
        maxcv = max(maxcv, rcc.max())
        covs.append(("cov($R_{%i,%i}$)" % (i, j), rcc, models[i]))
        rcm = np.mean(d[n]['_'], 1)
        l = plt.plot(rcm, label="mean($R_{%i, %i}$)" % (i, j))
    plt.legend()
    for i, c in enumerate(covs):
        plt.subplot(2, nk, nk + i + 1)
        plt.title(c[0])
        plt.imshow(c[1], vmin=mincv, vmax=maxcv, interpolation='nearest')
        b = c[2]
        plt.fill((b[0], b[0], b[1], b[1]), (b[0], b[1], b[1], b[0]), edgecolor='k',
                 facecolor='none')
        plt.xlim([0, c[1].shape[1]])
        plt.ylim([0, c[1].shape[0]])
    savefig(d['label'], 'Conditionals', fig)


def describe(d, figs=(1, 1, 1)):
    '''
    Create a canonical figure describing an experiment. d is a document as
    returned by runSystems. labl is an experiment label. If It is 1 character
    (eg "A", "B") then it is appended to the current date. If it is several
    characters, it is used as is.

    Comment is an arbitrary string that appears as text in the upper right of
    the figure, after the experiment label.

    If you are running interactive with -pyplot, then this figure is displayed.
    It is also saved to a png file on your desktop. The file save will also
    occur if called from a script.
    '''
    describeS(d, figs[0])
    describeM(d, figs[1])
    describeE(d, figs[2])


def testme(d):
    return len(d.findall(gd.search.ARRAY))


def cSpace(d, ce, ue, cmode='istac', clevel=3):
    ceisgrp = type(ce) in [tuple, list]
    ueisgrp = type(ue) in [tuple, list]
    comp = DRMODES[cmode]
    nc = _keys(d, 'c')
    cd = gd.Doc()
    if ceisgrp:
        cea = eqrc([d[n]['_'] for n in ce])
    else:
        cea = d[ce]['_']
    if ueisgrp:
        uea = eqrc([d[n]['_'] for n in ue])
    else:
        uea = d[ue]['_']
    iss = comp(cea, uea, clevel)
    n = len(nc)
    cn = "c%i" % n
    cd[cn + '._'] = iss.transpose()
    if ceisgrp:
        cd[cn + '.primary'] = "union%s" % (str(tuple(ce)),)
    else:
        cd[cn + '.primary'] = ce
    cd[cn + '.method'] = cmode
    cd[cn + '.order'] = clevel
    if ueisgrp:
        cd[cn + '.reference'] = "union%s" % (str(tuple(ue)),)
    else:
        cd[cn + '.reference'] = ue
    return cd


def pltFirstStac(d, labl='A', mode='istac', clevel=3):
    '''
    Generate a figure containing the basic pairwise compressions between the ensembles in
    an experiment represented by document d. "labl" is as in "describe". "mode" is a
    key into the ditionary ll.DRMODES and specifies the mode of dimensionality reduction.

    clevel is the parameter that determines the size of the reduced spaces. If an integer,
    this is a number of dimensions. Otherwise, its meaning depends on the mode.

    '''
    if len(labl) == 1:
        labl = time.strftime("%m%d%y") + labl
    f = plt.figure(1)
    plt.clf()
    plt.figtext(.1, .97, "Experiment %s" % labl)
    plt.subplot(511)
    c = cSpace(d, 'u0', 'u1', mode, clevel)
    plt.title("C(U0, U1, %s, %.2g)" % (mode, clevel))
    plt.imshow(c['c0._'].transpose(), aspect=3, interpolation='nearest')
    plt.subplot(512)
    c = cSpace(d, 'r0_0', 'r0_1', mode, clevel)
    plt.title("C(R0, R1, %s, %.2g)" % (mode, clevel))
    plt.imshow(c['c0._'].transpose(), aspect=3, interpolation='nearest')
    plt.subplot(513)
    c = cSpace(d, 'u0', 'r0_0', mode, clevel)
    plt.title("C(U0, R0, %s, %.2g)" % (mode, clevel))
    plt.imshow(c['c0._'].transpose(), aspect=3, interpolation='nearest')
    plt.subplot(514)
    c = cSpace(d, 'u1', 'r0_1', mode, clevel)
    plt.title("C(U1, R1, %s, %.2g)" % (mode, clevel))
    plt.imshow(c['c0._'].transpose(), aspect=3, interpolation='nearest')
    plt.subplot(515)
    c = cSpace(d, ['u0', 'u1'], ['r0_0', 'r0_1'], mode, clevel)
    plt.title("C( (U0, U1), (R0, R1), %s, %.2g)" % (mode, clevel))
    plt.imshow(c['c0._'].transpose(), aspect=3, interpolation='nearest')
    #f.set_figheight(8)
    f.subplots_adjust(left=.02, right=.99, bottom=.03, top=.93, wspace=.1, hspace=.5)

    f.canvas.draw()
    figname = "E" + labl + 'CompressionsWith_%s.png' % mode
    figname = os.path.join(FIGDIR, figname)
    plt.savefig(figname, format='png')


def jspaceVert(d, cmode='istac', clevel=3):
    d = d.fuse(cSpace(d, 'r0_0', 'u0', cmode, clevel))
    d = d.fuse(cSpace(d, 'r0_1', 'u1', cmode, clevel))
    d['c3.raw'] = np.column_stack([d['c0._'], d['c1._']])
    d['c3.orth'] = com.sla.orth(d['c3.raw'])
    return d


def sndStac(d2, reverse=False):
    ref = np.dot(d2['c3.orth'].transpose(), d2['r0_0._'])
    pri = np.dot(d2['c3.orth'].transpose(), d2['r0_1._'])
    iss = istac.istacspace(pri, ref, 0, reverse)
    iss['full'] = np.dot(d2['c3.orth'], iss['vecs'])
    return iss


def pltSndStac(d, labl='A', mode='istac', clevel=3, reverse=False):
    '''
    Generate a secondary compressed space between the RCSEs in a
    two-stimulus-one-system experiment described by document d.
    labl, mode, and clevel are as described for 'pltFirstStac'. Mode and
    clevel apply to the _first_ set of compressions. The second set is
    allways iSTAC-based, and recovers full dimension (clevel*2), but
    will use reverse iSTAC if "reverse" is True.


    '''

    iss = sndStac(jspaceVert(d, mode, clevel), reverse)
    f = plt.figure(1)
    plt.clf()
    plt.subplot(211)
    plt.plot(np.arange(1, len(iss['vals']) + 1), iss['vals'])
    if len(labl) == 1:
        labl = time.strftime("%m%d%y") + labl
    plt.figtext(.1, .97, "Experiment %s" % labl)
    if reverse:
        labl = labl + "REVERSE"
        plt.title('Retained KL vs # of coordinates in secondary REVERSE iSTAC space')
    else:
        plt.title('Retained KL vs # of coordinates in secondary iSTAC space')
    plt.subplot(212)
    plt.imshow(iss['full'].transpose(), aspect=3, interpolation='nearest')
    plt.title('Vectors in secondary iSTAC space, Highest KL retained on top')
    f.canvas.draw()
    figname = "E" + labl + 'SecondaryCompressionCondWith_%s.png' % mode
    figname = os.path.join(FIGDIR, figname)
    plt.savefig(figname, format='png')
    return iss


# FIXME: Functions after here don't support the v3 Doc instances yet. 
def llFig(d, compress='istac', clevel=.85):
    cp = d.sub('cpars').fuse({'compress': compress, 'clevel': clevel})
    dr = compare2(d, cp)[0]
    f = plt.figure(1)
    plt.clf()
    plt.subplot(111)
    paper._llplt(dr['fitness.raw'])
    f.subplots_adjust(left=.05, right=.98, bottom=.1, top=.98, wspace=.05)
    f.canvas.draw()


def _measure(d):
    ll = d['fitness.raw']
    c12 = (ll[0, 0, 0] - ll[0, 1, 0]) / (ll[0, 0, 1] + ll[0, 1, 1])
    c21 = (ll[1, 1, 0] - ll[1, 0, 0]) / (ll[1, 0, 1] + ll[1, 1, 1])
    return (c12, c21)


def cscan(d, clr=(1, 10), cmodes=('istac', 'pca', 'ppca', 'rand')):
    cp = d.sub('cpars').fuse({'compress': 'no', 'clevel': 0})
    cl0 = _measure(compare2(d, cp)[0])
    series = dict([(n, ([], [])) for n in cmodes])
    levels = apply(range, clr)
    for mode in cmodes:
        cp['compress'] = mode
        for clevel in levels:
            cp['clevel'] = clevel
            cl = _measure(compare2(d, cp)[0])
            series[mode][0].append(cl[0])
            series[mode][1].append(cl[1])
        series[mode][0].append(cl0[0])
        series[mode][1].append(cl0[1])
    return series


def pltcscan(series):
    cmodes = sorted(series)
    f = plt.figure(1)
    plt.clf()
    lines = {}
    i = 0
    for i, s in enumerate(cmodes):
        ser = series[s]
        lines["%s 1->2" % s] = plt.plot(ser[0], COLORS[i] + LSTYLES[0])
        lines["%s 2->1" % s] = plt.plot(ser[1], COLORS[i] + LSTYLES[1])
    n = sorted(lines)
    plt.figlegend([lines[k] for k in n], n, 'upper right')
    f.canvas.draw()


def pltcsp(css):
    f = plt.figure(1)
    plt.clf()
    for i, k in enumerate(['uc', 'rc', 'c1', 'c2', 'j']):
        plt.subplot(5, 1, i + 1)
        plt.imshow(css[k], aspect=3)
    f.canvas.draw()


def _splt(iss, sig=.05):
    splt = 0
    kl = 0
    vals = iss['vals']
    targ = iss['maxKL'] * (1 - sig)
    while kl < targ:
        if splt >= len(vals):
            break
        kl = vals[splt]
        splt += 1
    return splt


def pltpowloc(iss, rng=(32, 40)):
    #FIXME: automate
    vecs = iss['full']
    splt = _splt(iss)
    v1 = np.abs(vecs[:, :splt]).mean(1)
    v2 = np.abs(vecs[:, splt:]).mean(1)
    f = plt.figure(1)
    plt.clf()
    plt.plot(v1)
    plt.plot(v2)
    plt.axvspan(rng[0], rng[1], alpha=.3)
    plt.title("Power in components 0 to %i (blue), and %i to %i (green)" % (splt - 1, splt, vecs.shape[1]))
    f.canvas.draw()


def plotprojiss(d, iss, us=True):
    vecs = iss['full'].transpose()
    rs = _keys(d, "r")
    if us:
        us = _keys(d, "u")
    else:
        us = []
    #	splt = _splt(iss)
    #	if splt == 0 or splt >=vecs.shape[1]:
    #		report('This wont work. Space splits into a 0 witdth and a full width subspace')
    #		return
    #	v1 = vecs[:,:splt].transpose()
    #	v2 = vecs[:,splt:].transpose()
    f = plt.figure(1)
    plt.clf()
    for r in us + rs:
        r = r[0]
        rdat = d[r]['_'][:, :1000]
        pts = np.dot(vecs, rdat)
        plt.subplot(121)
        plt.plot(pts[0, :], pts[1, :], '.', label=r)
        plt.subplot(122)
        plt.plot(pts[-2, :], pts[-1, :], '.', label=r)
    plt.subplot(121)
    plt.title('Highest 2 KL')
    plt.legend()
    plt.subplot(122)
    plt.title('Lowest 2 KL')
    plt.legend()

    f.canvas.draw()


def plotprojiss3(d, iss):
    from mpl_toolkits.mplot3d import Axes3D

    vecs = iss['full'].transpose()
    rs = _keys(d, "r")
    us = _keys(d, "u")
    #	splt = _splt(iss)
    #	if splt == 0 or splt >=vecs.shape[1]:
    #		report('This wont work. Space splits into a 0 witdth and a full width subspace')
    #		return
    #	v1 = vecs[:,:splt].transpose()
    #	v2 = vecs[:,splt:].transpose()
    f = plt.figure(1)
    plt.clf()
    ax1 = f.add_subplot(121, projection='3d')
    ax2 = f.add_subplot(122, projection='3d')
    c = ['b', 'r', 'g', 'k', 'c', 'y', 'm']
    m = ['o', '^', '*']
    for i, r in enumerate(us + rs):
        r = r[0]
        rdat = d[r]['_'][:, :1000]
        pts = np.dot(vecs, rdat)
        ax1.scatter(pts[0, :], pts[1, :], pts[2, :], c=c[i], marker='o', label='%s in high KL' % r)
        ax2.scatter(pts[-1, :], pts[-2, :], pts[-3, :], c=c[i], marker='o', label='%s in low KL' % r)
    #ax.legend()
    plt.subplot(121)
    plt.title('Highest 3 KL')
    plt.subplot(122)
    plt.title('Lowest 3 KL')
    f.canvas.draw()
	
	
