#!/usr/bin/env python -3
# encoding: utf-8

#Created by Graham Cummins on Jun 7, 2011

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

"""
This module is a clone of selectUI, made on Feb 27 2012. This module is
accumulating a bunch of failed experiments, which should be kept accessible
for referencing until published, but which are not useful in the current
analysis runs, and cluttern the namespace. These include all methods of graph
comparison up to and including the "Droptree" ZSS functions. The various
commented-out tree comparison measures are rejected principally because they
are "home rolled", and wolud require more justification than the Edit
distance. The Droptree variants suffer from the problem that cross-validation
effects are pretty well washed out by the opperations of binning the
responses, clustering, and taking a consensus tree. To see why, consider a
case where stimulus A gives a reliable singlet, stimuli B and C give no
response, and stimulus D also gives no (true) response, but includes a random
singlet in one presentation. The True partition should be (A), (B, C, D), but
it is likely to get (A, D), (B, C). On a given cross-validation trial, this
clustering this cluster will only change if the one trial with the random
spike is dropped (these effects results from hierarchial clustering
considering only about the ordering of the distances, not the magnitude). A
change in the consensus tree will only occur if the ordering changes in half
or more of the cross validation trials, which is improbable (unless the number 
of dropped samples is >> half the total number of samples).

"""

"""
Functions for highlevel project control on the stimulus selectivity project.

This now also includes several functions intended for comparison and
visualization of trees, which derive from clustering of inter-class distance
matrixes.


tags:

CellExp from timingUI
Tree from clust
Roc from classify

(*) Clearly, all extension packages should define a single tags.py with all 
the tags used in the package. It avoids deeply nested nomenclature, and avoids
having to declare tag sharing, which is very common. 

(**) By convention I'm using "t" for boolean. The API in gicdat search 
includes boolean as "i" (integer), so the full spec would be 
i::self in [0,1]. This is fine, as far as the range of allowed data are 
concerned, but too slow, in the context of function signatures, which 
discriminate boolean flags from integers all the time. 

CellDoc: {KeySet('cell'):CellExp}
	
	This document contains keys such as "cell50", refferencing CellExp
	sub-documents. The sub documents contain compatible stimuli, meaning that 
	the same set of stimuli is used in each CellExp, and that the same stimulus
	id (integer) is always used to reffer to the same actual stimuls (waveform). 
	The keys are typically discontinuous (e.g. cell32 and cell35 may be included,
	but this doesn't imply that there is a cell33). 
	
	This module assumes that all CellExp in a CellDoc define conditions "cond1"
	and "cond2", and if any others are defined they will usually be ignored.
	
	This tag is the return type of timingUI.celldoc

CondName(d): Key(d, IntIO) (***)

	A clear type pattern which should be added to the spec is Key(d, p?).
	Key(d) are items in the set of keys of dictionary d. If the second 
	parameter "p" is given, then a Key(d, p) is a key that is in d, such that 
	d[p] matches pattern p. In this case, a CondName(d) is a key into Doc d 
	that reffers to an IntIO subdocument (eg. "cond1" or "cond2" in most of our
	experiments)


StimClass(S): Abstract

	The set of all responses to a particular (enumerated) stimulus condition.
	In the context of an IntIO, this could be represented as a partition or 
	a list of integers, e.g.
	[i for i in range(len(io['stims']) if  io['stims'][i] == S ]
	In this module, we use Stimulus S to be a string, which is the human-readable
	name of the stimulus file, rather than the enumeration index.

DistMode: s (****)

	Type "s" here is poor. It is passed as the argument "mode" to dist.dist but
	there is no type signature for that argument currently provided in module
	dist. It is an enumeration, since it needs to key dictionaries in module
	dist, but it isn't exactly a Key(dist.DMATS), or similar, because the
	function dist.dist parses the string if it contains an "_". It is exactly:
	Key(dist.DMATS) | Key(dist.VDISTS)+"_"+Key(dist.VREPS) I'm not sure how to
	conscisly represent that idea yet. Currently, the exhaustive enumeration is:
	vd, vdps, vdpe, ed_raw, ed_bin, ed_ist, ed_isi, ed_irate, cb_bin, cb_ist
	cb_isi, cb_irate of these vd, vdps, ed_ist, and ed_bin seem to be useful.
	vdpe (the expected poisson distance correction to the victor distance) would
	be useful, but the implementation in dist isn't complete yet.

DistQ: x

	The precision parameter used by some distance measures. Interprettation 
	depends on a DistMode. For example, in binning measures, this is the 
	bin width in microseconds. In Victor measures, this is 2.0/Vq, where Vq
	is Victor's "q" parameter - the cost of shifting an event by 1 microsecond.
	This also has the interprettation of being the longest distance by which a
	spike will ever be translated during the Victor transformation sequence. 

ICD: (dm of DistMat(N), names of N-[ of s)

	dm[i,j] is the expected distance between StimClass(s1) and Stimclass(s2)
	under some distance measure (a fully dependent version of the type would be
	ICDD(d: IntIO, m: DistMode, p: DistQ). The expected distance is the average
	of all pairwise distances between r_i, r_j where r_i is in StimClass(s1) and
	r_j in StimClass(s2). If s1==s2 the 0 distances between a response and
	itself are not considered in the average.
	
DirName: s
	Specifies the name of a directory in the file system. It will always be
	passed through os.path.expanduser beofore being used, meaning that "~" can
	be used to represent the current user's home directory. 

"""

import matplotlib.pyplot as plt
import gicdat.doc as gd
import gicdat.io as gio
import numpy as np
from timingUI import load, celldoc, preproc, showcell
from timingUI import rfromstim
from gicdat.enc import flat
import dist, vis, clust, os
from classify import roc
import tests


def stimnames(d):
    astim = set()
    for k in d:
        if k.startswith('cond'):
            astim = astim.union(set(d[k]['stims']))
    mas = max(tuple(astim))
    snames = []
    sc = d['stimclasses'] or {}
    for i in range(mas + 1):
        if "stim%i" % i in sc:
            snames.append(sc["stim%i" % i]['file'].split('.')[0])
        else:
            snames.append("Unknown Stim")
    return snames


def maxspikes(cells):
    """
    cells: CellDoc -> i

    return the largest number of spikes that result from any stimulus
    presentation (across all cells and conditions 1, 2) in the CellDoc.
    """
    cd = gd.Doc()
    mns = 0
    for cn in cells:
        for cond in ['cond1', 'cond2']:
            evts = cells[cn][cond]['evts']
            mls = max([len(e) for e in evts])
            mns = max(mns, mls)
    return mns


def snhist(cells):
    """
    cells:CellDoc -> hists: {cond1:N-#, cond2:N-#, cond1n:i, cond2n:i}

    Returns a dictionary of histograms. The keys cond# conatain arrays A such
    that A[i] is the number of presentations (of any stimulus, in any cell, in
    condition cond#) that resulted in i spikes. cond#n is the total number of
    presentations (of any stimulus in any cell in condition cond#)

    """
    mns = maxspikes(cells)
    hists = {'cond1': np.zeros(mns + 1), 'cond2': np.zeros(mns + 1),
             'cond1n': 0, 'cond2n': 0}
    for cn in cells:
        for cond in ['cond1', 'cond2']:
            evts = cells[cn][cond]['evts']
            for e in evts:
                ns = len(e)
                hists[cond][ns] += 1
                hists[cond + 'n'] += 1
    return hists


def rrps(cells):
    """
    cells: CellDoc -> (cns:N-[ of s, sns:N-[ of s,
                        hists:{cond1 cond2:N,N-# of x})

    cns is a list of cell names, sns is a list of stimulus names, hists is
    a dictionary of arrays such that hists[c][i,j] is the fraction of total
    responses of cell cns[i] in condition c which occured in respons to stimulus
    sns[j].

    """
    cns = cells.keys(0, 'cell', sort=True)
    stims = stimnames(cells[cns[0]])
    hists = {'cond1': np.zeros((len(cns), len(stims))),
             'cond2': np.zeros((len(cns), len(stims)))}
    for i, cn in enumerate(cns):
        for cond in ['cond1', 'cond2']:
            nts = len(flat(cells[cn][cond]['evts']))
            if not nts:
                continue
            for sno, sname in enumerate(stims):
                evts = rfromstim(cells[cn], cond, sno)
                if evts:
                    rr = float(len(flat(evts))) / float(nts)
                else:
                    rr = 0
                hists[cond][i, sno] = rr
            #hists[cond][i,:] = np.argsort(hists[cond][i,:])
    return (cns, stims, hists)


def show_rrps(cells):
    """
    cells: CellDoc -> None (draws in mpl figure 1)

    calls rrps(cells), and displays the result, using proportions of response
    spikes (that is, each row in the figure has the total height of all
    bars in the same condition sum to 1). Blue is condition 1, Red, condition 2

    """
    cns, sns, h = rrps(cells)
    f = plt.figure(1)
    plt.clf()
    x = np.arange(len(sns))
    mr = max(h['cond1'].max(), h['cond2'].max())
    for i in range(len(cns)):
        c1d = h['cond1'][i, :]
        c2d = h['cond2'][i, :]
        lmr = max(c1d.max(), c2d.max())
        if lmr != 0:
            c1d = .8 * c1d / lmr
            c2d = .8 * c2d / lmr
        plt.bar(x, c1d, color='b', bottom=i, width=.9)
        plt.bar(x, c2d, color='r', bottom=i, width=.6)
    plt.yticks(np.arange(len(cns)), cns)
    plt.xticks(np.arange(len(sns)) + .5, sns)
    plt.ylim([-.2, len(cns) + .2])
    f.canvas.draw()


def show_snhist(cells):
    """
    cells: CellDoc -> None (draws in mpl figure 1)

    Calls snhist(cells), and draws a bar chart of the results. This is the
    histogram of response rates for experimental conditions 1 (blue) and 2 (red)

    """
    h = snhist(cells)
    f = plt.figure(1)
    plt.clf()
    x = np.arange(len(h['cond1']))
    plt.bar(x, h['cond1'] / float(h['cond1n']), color='b', label='control')
    plt.bar(x, h['cond2'] / float(h['cond2n']), color='r',
            width=.5, label='Bic/Strych')
    plt.xlabel('Number of spikes')
    plt.ylabel('Fraction of responses')
    plt.legend(loc='best')
    plt.xlim([0, 30])
    plt.title('Response rates with and without inhibition')
    f.canvas.draw()


def allrasters(cells=(), save="~/Dropbox/wsu/cp_ic_rasters"):
    """
    cells: [ of i, save: DirName -> None (writes many files, draws in MPL
        figure 1)

    Differs from the function of the same name in timingUI in that it calls
    celldoc(cells) to load data, which means that only the common stimuli are
    included

    """
    save = os.path.expanduser(save)
    d = celldoc(cells)
    for cn in d:
        showcell(d[cn], save=os.path.join(save, "%s.png" % cn))


def dist_discrim_incond(d, mode='vdps', q=6000, nstim=None):
    """
    d: IntIO, mode: DistMode, q: x ->
        ret: ICD(d[cond], mode, q)

    norm is a flag. If True, the inter-class distances are normalized by the
    expected inclass distance:

    """
    dm = dist.dist(d['evts'], mode, q)
    dc = dist.classDM(dm, d['stims'], nstim)
    dca = np.zeros((len(dc), len(dc)))
    for i in range(len(dc)):
        for j in range(len(dc[i])):
            dca[i, j] = np.mean(dc[i][j])
            if i != j:
                dca[j, i] = dca[i, j]
    return dca


def _draw_icd(a, labs, vmin=None, vmax=None):
    """Internal, used by show_icd and friends"""
    plt.imshow(a[::-1, :], interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.colorbar()
    ll = ["%s (%i)" % (labs[i], i + 1) for i in range(len(labs))]
    plt.xticks(range(len(labs)), ["%i" % (i + 1,) for i in range(len(labs))])
    plt.yticks(range(len(labs))[::-1], ll)


def show_icd(dm, labs, fig=1):
    '''
    dm: DistMat(N), labs: N-[ of s, fig: i -> None (draws in matplotlib figure(fig) )

    Plots the distance matrix dm as an image in figure fig, labeling the rows
    with the strings in labs.

    '''
    f = plt.figure(fig)
    plt.clf()
    _draw_icd(dm, labs)
    f.canvas.draw()


def _gettreelinks(t):
    """Internal, used by show_icd_ld"""''
    nn = len(t) + 1
    ld = np.zeros((nn, nn))
    for i in range(1, nn):
        for j in range(i):
            d = clust.treedist_links(t, i, j)
            ld[i, j] = d
            ld[j, i] = d
    return ld


def show_icd_ld(dm, labs, fig=1):
    """
    Like show_icd, but instead of showing the inter-class distance matrix it
    shows the graph link distance. In this matrix, the i,j element is the
    number of edges that must be followed to get from i to j in the DAG
    generated by median-distance heirarchial clustering. It is thus in one
    sense an ordinal version of the raw distance matrix.
    """
    f = plt.figure(fig)
    plt.clf()
    t = clust.dtree(dm)
    tldm = _gettreelinks(t)
    _draw_icd(tldm, labs)
    f.canvas.draw()


def _tsort(a, l):
    """
    a: DistMat(N), l: N-[ of s -> DistMat(N), N-[ of s

    creates a new matrix and list of names sorted using tree clustering
    """

    t = clust.dtree(a)
    co = clust.treesort(t)
    l2 = [l[i] for i in co]
    a2 = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a2[i, j] = a[co[i], co[j]]
    return (a2, l2)


def rcell(cell):
    """
    cell:i | CellExp -> (CellExp, s)

    convert cell input which could be a document or a cell id integer to a
    document and name
    """
    if type(cell) == int:
        d = load(cell)
        cn = "cell%i" % cell
    else:
        d = cell
        try:
            cn = "cell%i" % (d['docsource.cell'],)
        except TypeError:
            cn = d['docsource.cell']
    return (d, cn)


def show_cell_icd(cell, mode='vdps', q=6000, fig=1,
                  save="~/Dropbox/wsu/roc/icd", suf='', tsort=1,
                  cn=None):
    """
    cell: i, mode: DistMode, q: DistQ, fig: i, save: DirName, suf: s, tsort:i
        -> None (draws in matplotlib figure fig, may write a file)

    Shows the inter-class distances for each stimulus in each condition, and
    also the clustering diagram. Draws a figure. If save is non-empty, then
    also writes a file in that directory, named:
    "c%i_icd_%s%i%s.png" % (cell, mode, int(q),suf)

    tsort is a flag. If true, sorts the dm using clust.treesort
    """
    d, cn = rcell(cell)
    f = plt.figure(fig)
    plt.clf()
    plt.figtext(0, .98, "%s, mode=%s, q=%.2g" %
                        (cn, mode, q), backgroundcolor='white')
    plt.subplot(121)
    labs = stimnames(d)
    a1 = dist_discrim_incond(d['cond1'], mode, q, len(labs))
    a2 = dist_discrim_incond(d['cond2'], mode, q, len(labs))
    vmin = min([a1.min(), a2.min()])
    vmax = max([a1.max(), a2.max()])
    if tsort:
        olabs = labs
        a1, labs = _tsort(a1, labs)
    _draw_icd(a1, labs, vmin, vmax)
    plt.title("Inter-class distances, Condition 1")
    plt.subplot(122)
    plt.title("Inter-class distances, Condition 2")
    if tsort:
        if tsort > 1:
            a2, labs = _tsort(a2, olabs)
        else:
            inds = [olabs.index(l) for l in labs]
            na2 = np.zeros_like(a2)
            for i in range(a2.shape[0]):
                for j in range(a2.shape[1]):
                    na2[i, j] = a2[inds[i], inds[j]]
            a2 = na2
    _draw_icd(a2, labs, vmin, vmax)
    f.canvas.draw()
    if save:
        save = os.path.expanduser(save)
        fn = os.path.join(save, "%s_icd_%s%i%s.png" % (cn, mode, int(q),
                                                       suf))
        plt.savefig(fn)


def show_tree(t, labs=None, fig=1):
    """
    t: Tree(N) | s, labs: N-[ of s, fig: i -> None (draws in MPL figure fig)

    Plots an image of tree t made with graphviz. This is a display wrapper for
    vis.dot2img and clust.tree2dot

    If t is a string to start with, it should be in the dot language (so this
    function can be used to plot other types of trees than those made by
    tree2dot). In this case labs has no effect

    """
    if not type(t) in [str, unicode]:
        t = clust.tree2dot(t, labs)
    f = plt.figure(fig)
    plt.clf()
    im = vis.dot2img(t)
    plt.imshow(im)
    a = f.get_axes()[0]
    a.yaxis.set_visible(False)
    a.xaxis.set_visible(False)
    f.canvas.draw()


def show_ctree(ct, labs=None, fig=1):
    if labs == None:
        labs = max([max(c) for c in ct[0]]) + 1
    if type(labs) == int:
        labs = map(str, range(labs))
    s = clust.ctree2dot(ct[0], labs, ct[1])
    show_tree(s, fig=fig)


def show_cell_clust(cell, mode='vdps', q=6000, fig=1,
                    save="/home/gic/Dropbox/wsu/clust", suf='', cn=None):
    """
    cell: i, mode: DistMode, q: DistQ, fig: i, save: DirName, suf: s
        -> None (draws in matplotlib figure fig, may write a file)

    Shows the clustering tree (as show_tree) for each stimuli in each condition,
    and also the clustering diagram. Draws a figure. If save is non-empty, then
    also writes a file in that directory, named: "c%i_clust_%s%i%s.png" % (cell,
    mode, int(q),suf)

    """
    d, cn = rcell(cell)
    f = plt.figure(fig)
    plt.clf()
    plt.figtext(0, .98, "%s, mode=%s, q=%.2g" % (cn, mode, q), backgroundcolor='white')
    labs = stimnames(d)
    a1 = dist_discrim_incond(d['cond1'], mode, q, len(labs))
    a2 = dist_discrim_incond(d['cond2'], mode, q, len(labs))
    sb = plt.subplot(121)
    plt.title("Condition 1")
    im = vis.dot2img(clust.tree2dot(clust.dtree(a1), labs))
    plt.imshow(im)
    sb.yaxis.set_visible(False)
    sb.xaxis.set_visible(False)
    sb = plt.subplot(122)
    plt.title("Condition 2")
    im = vis.dot2img(clust.tree2dot(clust.dtree(a2), labs))
    plt.imshow(im)
    sb.yaxis.set_visible(False)
    sb.xaxis.set_visible(False)
    f.canvas.draw()
    if save:
        save = os.path.expanduser(save)
        fn = os.path.join(save, "%s_clust_%s%i%s.png" % (cn, mode, int(q),
                                                         suf))
        plt.savefig(fn)


def dropsamp(cond, ndrop):
    """
    cond: IntIO, ndrop: i -> IntIO

    Return a document like cond, but with ndrop samples removed at random

    """
    n = len(cond['stims'])
    dr = np.random.permutation(n)
    nd = gd.Doc({'stims': [], 'evts': []})
    for i in range(n - ndrop):
        nd['stims'].append(cond['stims'][dr[i]])
        nd['evts'].append(cond['evts'][dr[i]])
    return nd


def drop_tree(cond, mode='vdps', q=6000, nbs=20,
              ndrop=1, N=14):
    trees = []
    for _ in range(nbs):
        d = dropsamp(cond, ndrop)
        dm = dist_discrim_incond(d, mode, q, N)
        if np.any(np.isnan(dm)):
            print('Warning: subsample changes classes. Discarding')
            continue
        trees.append(clust.dtree(dm))
    return trees


def bootstrap_tree(cond, mode='vdps', q=6000, nbs=20,
                   ndrop=1, show=0, names=None):
    """
    cond: IntIO, mode:DistMode, q: DistQ, nbs:i, ndrop:i ->

    """
    if not names:
        sids = tuple(sorted(set(cond['stims'])))
        names = ['s%i' % i for i in sids]
    trees = drop_tree(cond, mode, q, nbs, ndrop, len(names))
    ft, nr = clust.mrtree(trees)
    if show:
        s = clust.ctree2dot(ft, names, nr)
        show_tree(s, fig=show)
    return (ft, nr)


def noise_tree(cond, nbs=20, noise=.1, mode='vdps', q=6000, N=14):
    """
    Construct a consensus tree of trees generated by adding noise values
    to the distance matrix. These values are drawn from a N(0, sd) where
    sd is noise*dm.std

    """
    dm = dist_discrim_incond(cond, mode, q, N)
    nsd = noise * dm.std()
    trees = []
    for _ in range(nbs):
        dm2 = dm + np.random.normal(0, nsd, dm.shape)
        dm2 = np.maximum(dm2, 0)
        trees.append(clust.dtree(dm2))
    return clust.mrtree(trees)


def random_dm(n, minv=0, maxv=1):
    """
    n:N of i, minv:M of x, maxv:X of x :: X>M ->  N,N-#

    return a uniform random distance matrix on the indicated range (symetric,
    diagonal values are 0, off diagonal values are uniform random numbers
    """
    dm = np.random.uniform(minv, maxv, (n, n))
    for i in range(n):
        dm[i, i] = 0.0
        for j in range(i):
            dm[j, i] = dm[i, j]
    return dm


def random_bstrap_tree(mode='vdps', q=6000, nbs=20, ndrop=1, nstims=14,
                       npres=20, winlen=200000, nspikes=5):
    rt = nspikes / (float(winlen) / 1e6)
    d = tests.cdoc((tests.AC(winlen, rt),), nstims, npres)['condhpois']
    return bootstrap_tree(d, mode, q, nbs, ndrop, 0, None)


def tree_stability(d, mode='vdps', q=6000, nbs=20, ndrop=5):
    """
    d: CellExp
    """
    ts = []
    nd = d.copy()
    astim = d['cond1.stims'] + d['cond2.stims']
    aevts = d['cond1.evts'] + d['cond2.evts']
    names = stimnames(d)
    nd['condAll'] = {'stims': astim, 'evts': aevts}
    ts = []
    for i, c in enumerate(['cond1', 'cond2', 'condAll']):
        ft, nr = bootstrap_tree(nd[c], mode, q, nbs, ndrop, 0, names)
        ts.append((ft, nr))
    ss = [np.array(ts[i][1]).sum() for i in range(3)]
    sm = 2 * ss[2] / (ss[0] + ss[1])
    return (sm, ts)


def droptree_groups(cd, mode='vdps', q=6000, nbs=10, ndrop=5):
    """
    cd : CellDoc other args for drop_tree

    returns a document that contains a KeySet t of documents with "_" a tree2tup
    of a clustering tree, "cell" a cell name from cd, "cond" a condition name.
    These represent the result of running several sample drop clustering trials
    (as done by drop_tree), and labeling the resulting trees for cell and
    condition

    """
    cn = cd.keys(0, 'cell')
    conds = ['cond1', 'cond2']
    d = gd.Doc()
    names = [cd[cn[0]]['stimclasses'][k]['file'] for k in
             cd[cn[0]]['stimclasses'].keys(sort=True)]
    i = 1
    for cell in cn:
        for cond in conds:
            trees = drop_tree(cd[cell][cond], mode, q, nbs, ndrop, len(names))
            for t in trees:
                d['t%i' % i] = {'_': clust.tree2tup(t), 'cell': cell,
                                'cond': cond}
                i += 1
    d['names'] = names
    return d


def _pairwise(l, f):
    z = f(l[0], l[1])
    dm = {}
    for i in range(len(l) - 1):
        for j in range(i, len(l)):
            dm[(i, j)] = f(l[i], l[j])
    return dm


def _dmfrompw(pw):
    z = 0
    for k in pw:
        z = max(z, max(k))
    z += 1
    dm = np.zeros((z, z))
    for k in pw:
        dm[k[0], k[1]] = pw[k]
        dm[k[1], k[0]] = pw[k]
    return dm


def _grpstats(dtg, pw):
    # first index is cell is the same  (0, 1)
    # second is condition is the same (0, 1)
    zz = [[[], []],
          [[], []]]
    for k in pw:
        t1 = "t%i" % (k[0] + 1,)
        t2 = "t%i" % (k[1] + 1,)
        i = int(dtg[t1]['cell'] == dtg[t2]['cell'])
        j = int(dtg[t1]['cond'] == dtg[t2]['cond'])
        zz[i][j].append(pw[k])
    return zz


def intergroup_zss(dtg):
    """dtg is output from droptree_groups"""
    ks = dtg.keys(0, 't', sort=True)
    ts = [clust.tup2tree(dtg[k]['_']) for k in ks]
    pw = _pairwise(ts, lambda x, y: clust.zssdist(x, y, dtg['names']))
    zz = _grpstats(dtg, pw)
    return (zz, pw)


def show_intergroup_zss(zz, pw, dtg, nbs=10, nbars=10):
    f = plt.figure(1)
    plt.clf()
    plt.hist(zz[0][0], nbars, color='r', normed=True,
             label='different cell and condition')
    plt.hist(zz[1][1], nbars, color='b', normed=True,
             label='same cell and condition')
    plt.hist(zz[0][1], nbars, color=(1, .5, 0), normed=True,
             alpha=.6, label='different cell')
    plt.hist(zz[1][0], nbars, color='g', normed=True,
             alpha=.6, label='different condition')
    plt.legend(loc='best')
    f.canvas.draw()
    f = plt.figure(2)
    plt.clf()
    dm = _dmfrompw(pw)
    plt.imshow(dm)
    plt.colorbar()
    x = np.arange(0, dm.shape[1], nbs * 2) + nbs
    cn = [dtg['t%i.cell' % i] for i in x]
    plt.xticks(x, cn, rotation='vertical')
    f.canvas.draw()


def ctree_zss(cd, mode='vdps', q=6000, nbs=10, ndrop=5):
    """Calculates ZSS distance on Ctrees"""
    nd = gd.Doc()
    for cell in cd.keys(0, 'cell'):
        stn = stimnames(cd[cell])
        for cond in ['cond1', 'cond2']:
            (ft, nr) = bootstrap_tree(cd[cell][cond], mode, q, nbs, ndrop,
                                      0, stn)
            nd[cell + '.' + cond] = (ft, nr)
        try:
            nd[cell]['intercond_zss'] = clust.zssdist(nd[cell]['cond1'],
                                                      nd[cell]['cond2'], stn)
        except:
            print('failed')
            return nd[cell]
    for cell in nd.keys(0, 'cell'):
        icds = []
        for c2 in nd.keys(0, 'cell'):
            if c2 != cell:
                icds.append(clust.zssdist(nd[cell]['cond1'],
                                          nd[c2]['cond1'], stn))
                icds.append(clust.zssdist(nd[cell]['cond2'],
                                          nd[c2]['cond2'], stn))
        nd[cell]['icds'] = icds
    return nd


def show_ctree_zss(d):
    f = plt.figure(1)
    plt.clf()
    cells = d.keys(0, 'cell', sort=True)
    for i, k in enumerate(cells):
        plt.hist(d[k]['icds'], 8, bottom=i, color='b', normed=True)
        plt.vlines([d[k]['intercond_zss']], i, i + 1, lw=10)
    plt.yticks(range(len(cells)), cells)
    f.canvas.draw()


def full_intergroup_zss(cd, mode='vdps', q=6000, nbs=10, ndrop=5):
    dtg = droptree_groups(cd, mode, q, nbs, ndrop)
    zz, dm = intergroup_zss(dtg)
    show_intergroup_zss(zz, dm, dtg)
    return (zz, dm, dtg)

## This method of tree comparison is depricated in favor of ZSS 	
##def all_tree_stability(cd, mode='vdps', q=6000, nbs = 20, ndrop=5, show=1):
##	dname = os.path.expanduser("~/Dropbox/wsu/clust/stability")
##	suf = ''
##	od = gd.Doc()
##	for cn in cd:
##		if not cn.startswith('cell'):
##			continue
##		sm, ts = tree_stability(cd[cn], mode, q, nbs, ndrop)
##		od[cn] = {'stability':sm, 'ctrees':ts}
##		print('%s : %.3g' % (cn, sm))
##		if show:
##			names = stimnames(cd[cd.find('_', kp=':^cell', depth=1).next()])
##			for i, c in enumerate(['cond1', 'cond2', 'condAll']):
##				ft, nr = ts[i]
##				f=plt.figure(1)
##				plt.clf()
##				plt.title("Bootstrap Majority Tree, %s" % c)
##				s = clust.ctree2dot(ft, names, nr)
##				im = vis.dot2img(s)
##				plt.imshow(im)	
##				sb = f.get_axes()[0]
##				sb.yaxis.set_visible(False)
##				sb.xaxis.set_visible(False)
##				f.canvas.draw()
##				fn = os.path.join(dname, "%s_%s_mrtree%s%i%s.png" % (cn, c, mode,
##					                int(q), suf))
##				plt.savefig(fn)
##	return od
##
##def trees_across_cells(cd, mode='vdps', q=6000):
##	trees = []
##	names = None
##	for cn in cd:
##		if not cn.startswith('cell'):
##			continue
##		if names is None:
##			names = stimnames(cd[cn])
##		for cond in ['cond1', 'cond2']:
##			dm = dist_discrim_incond(cd[cn][cond], mode, q, len(names))	
##			trees.append(clust.dtree(dm))
##	ft, nr = clust.mrtree(trees)
##	s = clust.ctree2dot(ft, names, nr)
##	show_tree(s)
##	
##def tree_sim_per_q(cell, mode='vdps', qr = [1, 20000, 4000]):
##	pass
##
##def do_all_icd(cd, q = 6000, mode='vdps', save="/home/gic/Dropbox/wsu/clust", 
##               suf=''):
##	"""
##	cd: CellDoc
##	(q, mode, save, suf): as for show_cell_icd, cells: [ of i -> None 
##		(writes files and draws in MPL figure 1)
##	
##	Does show_cell_icd and show_cell_clust with the indicated (q, mode, save, 
##	suf) and fig=1, for every cell in cd. The point of this is to save 
##	all the image files. Drawing in figure 1 is rather useless, since it 
##	overwrites the things it draws often. Probably should be modified to
##	use a file based MPL backend instead, eventually.
##	
##	"""
##	for cell in cd:
##		show_cell_icd(cd[cell], mode, q, cn = cell)
##		show_cell_clust(cd[cell], mode, q, cn = cell)
##
##def icd_treediff(cell, mode, q):
##	"""
##	cell: i | CellExp, mode: DistMode, q: DistQ -> float
##	
##	"""
##	if type(cell) == int:
##		d = load(cell)
##	else:
##		d = cell
##	labs = stimnames(d)
##	a1= dist_discrim_incond(d['cond1'],mode, q, len(labs))
##	a2= dist_discrim_incond(d['cond2'],mode, q, len(labs))
##	t1 = clust.dtree(a1)
##	t2 = clust.dtree(a2)
##	td = clust.cmptrees(t1, t2)
##	return (td.sum() - np.diag(td).sum())/float(td.shape[0])
##
##def scan_cell_tdiff(cell, modes=['ed_bin', 'vd', 'vdps', 'ed_ist'], 
##                    qs = [1000, 2000, 5000, 10000, 20000, 100000],
##                    plot=True, save=''):
##	"""
##	cell: i, modes: [ of DistMode, qs: [ of DistQ, plot: t, save: FileName ->
##		N-M-# of x::N==len(modes), M==len(qs)    (may plot in MPL fig 1 
##		and may save a file)
##	
##	returns an array of the values of icd_treediff(cell, mode, q) for each 
##	mode in modes and each q in qs.
##	
##	if plot is true, draw a graph of the results, with one line per mode, and
##	one point per q. If save is non-empty, also save the plot
##	"""
##	d = load(cell)
##	out = np.zeros((len(modes), len(qs)))
##	if plot:
##		f = plt.figure(1)
##		plt.clf()
##		qs = np.array(qs)
##	for i in range(out.shape[0]):
##		for j in range(out.shape[1]):
##			out[i,j] = icd_treediff(cell, modes[i], qs[j])
##		if plot:
##			plt.plot(qs, out[i,:], label = modes[i])
##	if plot:
##		plt.legend(loc='best')
##		f.canvas.draw()
##		if save:
##			plt.savefig(save)
##	return out
##
##def scan_many_cell_tdiffs(cells=GOOD, 
##                          modes=['ed_bin', 'vd', 'vdps', 'ed_ist'], 
##                          qs = [1000, 2000, 5000, 10000, 20000, 100000],
##                          save='~/Dropbox/wsu/roc/tdiffs'):
##	"""
##	cells: [ of i, (modes, qs) as for scan_cell_tdiff, save: DirName -> 
##		None (writes files in save, draws in MPL figure 1)
##
##	Runs scan_cell_tdiff for each cell in cells, with a save argument of 
##	save/cell#.png
##	
##	"""
##	save = os.path.expanduser(save)
##	for cell in cells:
##		cn = "cell%i.png" % cell
##		scan_cell_tdiff(cell, modes, qs, True, os.path.join(save, cn))
##	
def pairwise_ROC(d, cond='cond1', mode='ed_bin', q=20000, nroc=100):
    """
    d: CellExp, cond: CondName(d), mode: DistMode, q: DistQ, nroc: i ->
        ret: {(i,i) of Roc(nroc)

    Constructs every pairwise comparison between stimuli in d[cond] as a
    classify.roc ROC curve. Returns a dictionary of pairs (i, j) onto Rocs,
    where (i, j) are the two stimuli being separated.
    """
    dm = dist.dist(d[cond]['evts'], mode, q)
    st = np.array(d[cond]['stims'])
    stn = _stimnames(d, cond, True)
    stpairs = clust.pairs(sorted(stn))
    ret = {}
    for p in stpairs:
        s1 = np.nonzero(st == stn[p[0]])[0]
        s2 = np.nonzero(st == stn[p[1]])[0]
        ind = np.union1d(s1, s2)
        scla = np.array([(x in s1 or 2) for x in ind])
        dm_p = dm[ind, :][:, ind]
        ret[p] = roc(scla, dm_p, nts=nroc)
    return ret


def show_pairROC(pr, fig=1):
    """
    pr: ret from pairwise_ROC, fig: i -> None (draws in MPL figure fig)

    Draws every curve in the pairwise_ROC scan pr. Usually rather useless.
    """
    f = plt.figure(fig)
    plt.clf()
    for k in pr:
        plt.plot(pr[k][:, 1], pr[k][:, 0], label=k[0] + '/' + k[1])
    plt.legend(loc='best')
    f.canvas.draw()


def _areaunder(a, npts=20):
    """
    a: Roc, npts: i -> x

    Estimates the area under the ROC curve a by resampling this curve
    uniformly onto npts samples and taking the sum (note that this doesn't
    account for the bin width, so you need to devide the result by npts to
    get a real area. It can be used to order ROCs by "goodness", though,
    provided all measurements are made with the same value of npts

    """
    aa = np.row_stack([np.array([[0, 0]]), a, np.array([[1, 1]])])
    x = np.unique(a[:, 1])
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i] = aa[np.nonzero(aa[:, 1] == x[i]), 0].max()
    aa = np.interp(np.linspace(0, 1, npts), x, y)
    return aa.sum()


def show_ordered_pairROC(pr, r=(0, 10), fig=1):
    """
    pr: ret from pairwise_ROC, r:(i, i), fig: i -> [ of (i,i) (draws a figure)

    Orders all the ROCs in pr according to the area under the curve. Plots the
    range range(r[0], r[1]) of curves from this ordering in MPL figure fig.
    Returns the list of pairs that correspond to the curves plotted.

    """

    def _cp(k1, k2):
        return cmp(_areaunder(pr[k2]), _areaunder(pr[k1]))

    ks = sorted(pr, cmp=_cp)
    ks = [ks[n] for n in apply(range, r)]
    #vs = [_areaunder(pr[k]) for k in ks]
    f = plt.figure(fig)
    plt.clf()
    for k in ks:
        p = plt.plot(pr[k][:, 1], pr[k][:, 0], label=k[0] + '/' + k[1], linewidth=3)
        c = p[0].get_color()
    #plt.fill_between( pr[k][:,1],0, pr[k][:,0], color=c)
    plt.legend(loc='best')
    f.canvas.draw()
    return ks

#### Deprecated clust


"""
Tools for clustering or grouping. Includes a frontend for PyCluster

tags: 

DistMat(N); N,N-# of x; Symetric matrix where element i,j is the distance
between two entities i and j under some distance metric or dissimilarity 
measure. The entities referenced by the rows (or columns) are refered to as 
items 

Partition1(N); N-[ of i; Collection of integers indicating group membership, 
such that if self[i] == self[j], items i and j are in the same group.

Partition(M,N); M-[ of [ of i:sum([len(l) for l in self])==N ; alternate 
syntax for expressing a partition of N items into M groups. self[i] is a 
list of the items that are in group i.

other types:

Tree(N); Pycluster.tree instance. List-like. Contains nodes. Each node has 
	attributes left, right, distance. Distance is the distance between clusters.
	left and right identify them. If these are >=0, they are item indexes. If 
	they are <0 they reffer to other clusters in the tree, such that -X is 
	node self[X-1]. A Tree(N) is the result of clustering N objects. The length
	of the tree is N-1.

Node; An integer, which refers to a node in a Tree, obeying the sign convention
	mentioned in Tree. Unless otherwise specified Node may be either a node or a
	leaf. Node(+) is used to mean strictly a leaf, and Node(-) to mean strictly
	a node.

"""

#TODO: There are too many types of "Tree" here, due to 3rd party libraries
#using different formats. Pycluster has a poor internal representation of
#trees, and I added a representation for consensus trees. ZhangShasha has a
#sensible "pythonic" form of tree (with Node objects that have any number of
#children), but this class doesn't provide an attribute for the probability
#of occurence used by consensus trees. The most sensible way to store trees
#would be as gd.Doc instances (which are, indeed, flexible trees already).
#Then the various methods such as "cut" and zssdist would need to be reworked
#to correctly convert the tree from Doc to appropriate form first and then
#apply the opperation, and the methods like dtree should return Doc. This
#would insulate client code (currently in timingInfo and selectUI) from the
#silly proliferation of trees. For now, though, I'll leave it as is I guess.


from __future__ import print_function, unicode_literals
import numpy as np
import gicdat.doc as gd
from gicdat.util import infdiag
from mixmod import mmcall

try:
    from Pycluster import treecluster, kmedoids, kcluster, somcluster, Tree, Node
except:
    from gicdat.control import report

    report("Pycluster is not found. Several methods in the clust module will fail")
try:
    from zss import compare as zscmp
    from zss.test_tree import Node as ZSNode
except:
    from gicdat.control import report

    report("Zhang-shasha is not found. pct2zst and zssdist in clust will fail")


def dt_first(dm, t):
    '''
    dm: DistMat(N), t: x -> ids: Partition1(N)

    A naive grouping function which groups dm using a threshold, t. It depends
    on the ordering of dm, since it assigns each item to the same group as the
    first other item with distance <=t.

    '''
    centers = []
    ids = []
    for id in range(dm.shape[0]):
        for c in centers:
            if dm[id, c] <= t:
                ids.append(c)
                break
        else:
            ids.append(id)
            centers.append(id)
    return ids


def dt_hclust(dm, t):
    '''
    dm: DistMat(N), t: x -> ids: Partition(M, N)

    Partitions dm using hclustmean, which is a local implementation of mean-
    distance heirarchical clustering. Stops when the maximum distance between
    clusters is <= t.

    '''
    memb = [[i] for i in range(dm.shape[0])]
    md = dm.max()
    while md > t:
        dm, memb = hclustmean(dm, memb)
    return memb


DTHRESH = {'first': dt_first,
           'hclust': dt_hclust,
}


def hclustmean(dm, memb):
    '''
    dm: DistMat(N), memb: Partition(N, L)  -> ndm: DistMat(N-1),
        newmemb: Partition(N-1, L)

    Performs one hierarchical clustering step using mean distance, based on the
    distance matrix dm and the membership array memb. Called internally by hca.

    '''
    dmi = infdiag(dm)
    closest = np.unravel_index(dmi.argmin(), dmi.shape)
    nm1 = len(memb[closest[0]])
    nm2 = len(memb[closest[1]])
    newc = memb[closest[0]] + memb[closest[1]]
    leaveout = [i for i in range(dm.shape[0]) if not i in closest]
    newmemb = [memb[i] for i in leaveout] + [newc]
    ndist = (dm[closest[0], :] * nm1 + dm[closest[1], :] * nm2) / (nm1 + nm2)
    ndm = np.zeros((len(newmemb), len(newmemb)), dm.dtype)
    ndm[-1, -1] = ndist[closest[0]]
    ndist = ndist[leaveout]
    ndm[-1, :-1] = ndist
    ndm[:-1, -1] = ndist
    ndm[:-1, :-1] = dm[leaveout, :][:, leaveout]
    return (ndm, newmemb)


def hca(dists, memb, nclust):
    '''
    dists: DistMat(N), memb: Partition(N, L), nclust: i
        -> (DistMat(nclust), Partition(nclust, L))

    Construct nclust clusters using agglomerative hierarchical clustering with
    mean distance. The inputs dists and memb specify an existing clustering. If
    memb is false, it defaults to [ [i] for i in range(dists.shape[0])] (which
    correponds to no initial clustering.

    This implementation is slower than dtree (which uses pycluster)
    '''
    if not memb:
        memb = [[i] for i in range(dists.shape[0])]
    while len(memb) > nclust:
        dists, memb = hclustmean(dists, memb)
    return dists, memb


def mixmodpartition(data, k, model="Gaussian_pk_Lk_Ck", reps=1):
    """Wraps mixmod.mmcal(data, [k], model, False, reps), and returns the
    resulting partition"""
    d = mmcall(data, [k], model, False, reps)
    return d['partition']

#dtree(dists, **cargs)		
#clust.cuttree(t, nclust)
#clust.mediods(dists, nclust, **cargs)
#clust.vtree(evts, **cargs)
#clust.kmeans(evts, nclust, **cargs)

#method: s, m, c, a    single-link, max, centroid, average 

def dtree(dm, method='a'):
    """
    dm: DistMat(N), method: s ('a') -> Tree(N)

    calls PyCluster.treecluster(method=method, distancematrix=dm), except that
    dm is copied, to protect from PyClusters INSANE behaviour of modifying dm
    in place.

    """
    #warning: pycluster writes the dm by side effect!!!!!
    dm = dm.copy()
    return treecluster(method=str(method), distancematrix=dm)


def vtree(dat, dist='e', method='a'):
    """
    dat: N,M-#, dist: s ('e'), method: s ('a') -> Tree(N)

    calls PyCluster.treecluster(dat, dist=dist, method=method),

    This require that dat contain data smaples in rows, and that the distance
    function is appropriate to use on the samples (e.g. the default dist='e' is
    the euclidean distance, and all PyCluster dist options are vector space
    distances).

    """
    #implicit transpose=0, aka samples are rows, and there is no mask or weight
    return treecluster(dat, dist=dist, method=str(method))

# clustering here seems to work, but it calls pycluster.distancematrix, which 
# normalizes in a strange way. In particular, selecting euclidean distance actually 
# gets you the square of the euclidean distance divided by the number of dimensions

#pycluster measures
#vspace measures:
#k means 
#data nclusters, mask, weight, transpose, npass, method, dist, initialid		
#mask weight transpose can probably be ommited. 

#dist mat measures:
#k-mediods
#distance, nclusters, npass, initialid
#tree
#data, mask, weight, transpose, method, dist, distancematrix

def tree2dot(tree, names=None):
    '''
    tree: Tree(N), names: [ of s (*) -> s

    (*) if False, defaults to ['n0', 'n1', ... 'nN']

    Makes a representation of a tree object in the graphviz dot language.

    '''
    from matplotlib.cm import get_cmap

    spectral = get_cmap('spectral')
    if not names:
        names = ["n%i" % (i, ) for i in range(len(tree) + 1)]
    gv = ['digraph G {']
    dists = [n.distance for n in tree]
    dmax = max(dists)
    dmin = min(dists)

    def _nn(j):
        if j >= 0:
            return names[j]
        else:
            return "c%i" % (-1 * j,)

    def _d2c(d):
        if dmin == dmax:
            return "#000000"
        d1 = (d - dmin) / (dmax - dmin)
        c = spectral(d1, bytes=True)[:3]
        s = "#%02x%02x%02x" % c
        return s

    for i in range(len(tree) - 1, -1, -1):
        nid = "c%i" % (i + 1,)
        lid = _nn(tree[i].left)
        rid = _nn(tree[i].right)
        dist = _d2c(tree[i].distance)
        gv.append('"%s" -> "%s" [penwidth=2, color="%s"];' % (nid, rid, dist))
        gv.append('"%s" -> "%s" [penwidth=2, color="%s"];' % (nid, lid, dist))
    gv.append('}')
    return "\n".join(gv)


def tree2tup(t):
    """Converts a Pycluster Tree to a list of tuples (which can be stored in a
    gd.Doc, repr/eval group, etc)"""
    return tuple([(n.left, n.right, n.distance) for n in t])


def tup2tree(t):
    """Restores a PyCluster Tree from the output of tree2tup"""
    return Tree([Node(*n) for n in t])


def treecontains(tree, node, leaf):
    '''
    tree: Tree(N), node: Node, leaf: Node -> bool

    True if node in tree contains leaf (recursively), False otherwise. Both node
    and leaf may be negative (node references) or non-negative (leaf
    references). If node is non-negative this is the recursion edge-case, and
    just tests node==leaf.

    '''
    if node >= 0:
        return node == leaf
    else:
        n = tree[-node - 1]
        return (treecontains(tree, n.left, leaf)
                or treecontains(tree, n.right, leaf))


def treebranch(tree, node):
    """
    tree: Tree, node: Node -> [ of i

    returns all the elements that are children of node in tree. If node is
    non-negative, this is the list [node], of corse.
    """
    if node >= 0:
        return [node]
    else:
        n = tree[-node - 1]
        return treebranch(tree, n.left) + treebranch(tree, n.right)


def treelevel(tree, node, leaf):
    '''
    tree: Tree(N), node: Node, leaf: Node -> i

    If treecontains(tree, node, leaf), the return value is the number of levels
    down from node that leaf is found. Otherwise, it is -1

    '''
    if node >= 0:
        if node == leaf:
            return 0
        else:
            return -1
    else:
        n = tree[-node - 1]
        d = treelevel(tree, n.left, leaf)
        if d == -1:
            d = treelevel(tree, n.right, leaf)
            if d == -1:
                return -1
        return d + 1


def treeparent(tree, node):
    """
    tree: Tree, node: Node -> None | Node(+)

    returns the node that is the direct parent of the input node. The input
    may be negative or non-negative. The return value will be negative (since
    a leaf is never a parent). If the specified node is the root (there is no
    parent) the return is None.

    """
    for i, n in enumerate(tree):
        if tree[i].left == node or tree[i].right == node:
            return -i - 1
    return None


def treedist_lev(tree, n1, n2):
    """
    tree: Tree, n1: Node, n2: Node -> i | inf

    Returns the number of levels up the tree you need to climb from n1 to reach
    a node that contains n2. Return is np.inf if one of the nodes is not in tree.

    Note that this is not a symettric function, since if one node is close to
    the root, all level distances will be very small (and also this is non-
    intuitive, since such a node is usually an outlier that was clustered last).
    """
    d = 0
    while not treecontains(tree, n1, n2):
        d += 1
        n1 = treeparent(tree, n1)
        if n1 == None:
            return np.inf
    return d


def treedist_levs(tree, n1, n2):
    """
    tree: Tree, n1: Node, n2: Node -> i | inf

    Average of treedist_lev(tree, n1, n2) and treedist_lev(tree, n2, n1)
    (thus symettric)

    """
    return (treedist_lev(tree, n1, n2) + treedist_lev(tree, n2, n1)) / 2.0


def treedist_links(tree, n1, n2):
    """
    tree: Tree, n1: Node, n2: Node -> i | inf

    Number of edges that need to be traversed to get from n1 to n2. This should
    be twice treedist_levs(tree, n1, n2), and somewhat faster to calculate.
    """
    d = 0
    tl = treelevel(tree, n1, n2)
    while tl == -1:
        d += 1
        n1 = treeparent(tree, n1)
        if n1 == None:
            return np.inf
        tl = treelevel(tree, n1, n2)
    return d + tl


def pairs(l):
    """
    l: [ of ? -> [ of 2-[ of ?

    returns a list of pairs containing all unique pairs of items in list l.
    This is a combination not a permutation, and also it assumes an item can't
    be paired with itself. Length of the return, if the input length is N,
    is thus N!/( 2*(N-2)! )
    """
    ret = []
    for i in range(1, len(l)):
        for j in range(i):
            ret.append((l[j], l[i]))
    return ret


def cmptrees(t1, t2):
    """
    t1:Tree(N), t2:Tree(N) ->op of M,M-# of i

    Compares two trees using treedist_links. This assumes that the trees
    connect the same nodes (e.g. that the semantics of the item indexes in the
    trees are the same), but only checks for the syntax (that the trees have the
    same length, and thus that the set of actual indexes is the same).

    The function checks all pairs of nodes, and compares their distance (in the
    links sense) in t2, and t2. The matrix op is a 2D histogram, such that
    op[i,j] specifies the number of pairs that had distance i+2 in t1, and
    distance j+2 in t2. The +2 is used to compress the matrix, since distances
    of 0 or trivial (node identity), and distance 1 is imposible under
    treedist_links, so 2 is the smallest distance of interest, and is listed in
    the first row/column. The size of op depends on how balanced the trees are.
    It is large enough to contain the largest distance that occurs (in either
    tree), which is on the range (log2(N), N+1).

    Identical trees will have only diagonal elements in op, so
    op.sum - np.diag(op).sum() gives a measure of dis-similarity between the
    trees.

    """
    if len(t1) != len(t2):
        raise ValueError("Can't compare trees of different sizes")
    n = len(t1)
    md = 0
    op = np.zeros((n + 2, n + 2))
    coords = []
    ps = pairs(range(n))
    for p in ps:
        d1 = treedist_links(t1, p[0], p[1])
        d2 = treedist_links(t2, p[0], p[1])
        op[int(d1), int(d2)] += 1
        md = max(md, max(d1, d2))
    return op[2:md + 1, 2:md + 1]


def branchlengths(t, n):
    """
    t: Tree, n: i -> (i, i)

    returns a tuple (min, max) of the lengths of the various subtrees of
    node n in tree t. If n is a leaf (n>=0), this is (0, 0). Otherwise, min is
    the smallest number of edges required to get from n to a leaf, and max is
    the largest number possible.

    """
    if n >= 0:
        return (0, 0)
    else:
        nid = -n - 1
        bll = branchlengths(t, t[nid].left)
        blr = branchlengths(t, t[nid].right)
        return (min(bll[0], blr[0]) + 1, max(bll[1], blr[1]) + 1)


def uplength(t, n, camefrom=None):
    """
    As branch lengths, but considers paths that go upward as well
    """
    if n >= 0:
        return (0, 0)
    else:
        nid = -n - 1
        milen = len(t) + 2
        malen = 0
        pn = treeparent(t, n)
        if pn and pn != camefrom:
            mi, ma = uplength(t, pn, n)
            milen = min(mi, milen)
            malen = max(ma, malen)
        if t[nid].left != camefrom:
            mi, ma = uplength(t, t[nid].left, n)
            milen = min(mi, milen)
            malen = max(ma, malen)
        if t[nid].right != camefrom:
            mi, ma = uplength(t, t[nid].right, n)
            milen = min(mi, milen)
            malen = max(ma, malen)
        return (milen + 1, malen + 1)


def _root(t):
    nodes = set([-n - 1 for n in range(len(t))])
    for n in t:
        nodes -= set([n.left, n.right])
    if len(nodes) == 1:
        return nodes.pop()
    return nodes


def treesort(t, lf=False, n=None):
    """
    t: Tree, lf: t->s: [ of i

    s is an ordering of the leafs in t (e.g. a list containing one copy each of
    every non-negative integer in t), which depends on the structure of t

    lf specifies whether to sort "longest first".

    """
    if n is None:
        n = _root(t)
        if type(n) == set:
            raise ValueError('use treesort_rootless for this tree')
    if n >= 0:
        return [n]
    nid = -n - 1
    nl = branchlengths(t, t[nid].left)
    nr = branchlengths(t, t[nid].right)
    lts = treesort(t, lf, t[nid].left)
    rts = treesort(t, lf, t[nid].right)
    if lf:
        if nl[1] < nr[1]:
            return rts + lts
        else:
            return lts + rts
    else:
        if nl[0] > nr[0]:
            return rts + lts
        else:
            return lts + rts


def tclusters(t):
    """
    t: Tree -> [ of [ of i

    Returns a partition list for the tree t
    """
    return [treebranch(t, -i - 1) for i in range(len(t))]


def compat(c1, c2):
    """
    c1:[ of i, c2:[ of i -> t

    are cluters c1 and c2 compatible (aka identical, disjoint, or nested)
    """
    c1 = set(c1)
    c2 = set(c2)
    if c1.isdisjoint(c2):
        return True
    elif c1.issubset(c2):
        return True
    elif c2.issubset(c1):
        return True
    return False


def contained_in(ft, ft2):
    inft2 = set()
    for f in ft2[0]:
        inft2 = inft2.union(f)
    new = []
    for f in ft[0]:
        nn = sorted(list(inft2.intersection(f)))
        if len(nn) > 1 and not nn in new:
            new.append(nn)
    return (new, ft[1])


def equal_clust_frac(ft1, ft2):
    n = 0
    for c in ft1:
        s1 = set(c)
        for c2 in ft2:
            if set(c2) == s1:
                n += 1
                break
    return float(n) / len(ft1)


def clustcounts(trees):
    atc = {}
    for t in trees:
        ttc = [tuple(sorted(s)) for s in tclusters(t) if len(s) <= len(t)]
        for c in ttc:
            if c in atc:
                atc[c] += 1
            else:
                atc[c] = 1
    return atc


def _tupsort(t1, t2):
    return cmp(len(t1), len(t2)) or cmp(t1, t2)


def mrtree(trees):
    atc = clustcounts(trees)
    ft = [k for k in atc if atc[k] > len(trees) / 2.0]
    nr = [float(atc[k]) / len(trees) for k in ft]
    return (ft, nr)


def _lchild(t, z):
    ii = -1
    ll = 0
    for i, tt in enumerate(z):
        if t.issuperset(tt) and len(t) > len(tt):
            l = len(tt)
            if l > ll:
                ii = i
                ll = l
    return ii


def buildtree(ft, n, nr=None):
    nn = [[set(range(n)), None, None, None]]
    if nr != None:
        nn[-1][-1] = 1.0
    active = [0]
    while active:
        ni = active.pop(0)
        lc = _lchild(nn[ni][0], ft)
        if lc == -1:
            pass
        #print('Warning: tree incomplete')
        else:
            if nr != None:
                prob = nr[lc]
            else:
                prob = None
            lset = set(ft[lc])
            rset = nn[ni][0].difference(lset)
            for i, s in enumerate([lset, rset]):
                if len(s) == 1:
                    nn[ni][i + 1] = s.pop()
                else:
                    id = -len(nn) - 1
                    nn[ni][i + 1] = id
                    if len(s) == 2:
                        st = tuple(s)
                        nn.append([s, st[0], st[1], prob])
                    else:
                        active.append(len(nn))
                        nn.append([s, None, None, prob])
    nodeids = {}
    return nn


def build_subset_tree(ctree):
    inft2 = set()
    for f in ft2:
        inft2 = inft2.union(f)
    new = []


def comparable_ctrees(t1, t2):
    t1c = contained_in(t1, t2)
    t2c = contained_in(t2, t1)


def ctree2dot(ft, names, nr=None):
    tree = buildtree(ft, len(names), nr)
    gv = ['digraph G {']
    for i in range(len(tree)):
        node = tree[i]
        nid = "c%i" % (i + 1,)
        c = tuple(sorted(node[0]))
        if node[-1] != None:
            gv.append('%s [label="%s ( %.2f )"];' % (nid, nid, node[-1]))
        if node[1] is None:
            for k in c:
                gv.append('"%s" -> "%s";' % (nid, names[k]))
        else:
            for v in [1, 2]:
                if node[v] < 0:
                    tn = "c%i" % (-1 * node[v],)
                else:
                    tn = names[node[v]]
                gv.append('"%s" -> "%s";' % (nid, tn))
    gv.append("}")
    return "\n".join(gv)


def clustcompatscore(ct1, ct2):
    """
    each ct is (ft, nr)

    """
    s = 0
    for i1, c1 in enumerate(ct1[0]):
        for i2, c2 in enumerate(ct2[0]):
            if compat(c1, c2):
                s += ct1[1][i1] + ct2[1][i2]
    return s

#zscmp ZSNode from zss

def pct2zst(t, names):
    """
    t: Tree -> ZSNode

    Convert a pycluster Tree into the tree type used by Zhang-shasha. The
    return type is a zss.test_tree.Node, containing the root of the tree.
    """
    r = t[-_root(t) - 1]
    rn = ZSNode('c')

    def addnode(t, i):
        if i >= 0:
            n = ZSNode(names[i])
        else:
            n = ZSNode('c')
            tn = t[-i - 1]
            for k in [tn.left, tn.right]:
                n.addkid(addnode(t, k))
        return n

    return addnode(t, _root(t))


def zst2dot(zt):
    """
    zt : ZSNode -> s

    Return a dot language representation of the ZSNode zt
    """
    gv = ['digraph G {']
    used = []

    def relabel(n):
        if n.label in used:
            i = 2
            while "%s%i" % (n.label, i) in used:
                i += 1
            n.label = "%s%i" % (n.label, i)
        used.append(n.label)

    def addnode(n):
        relabel(n)
        for k in n.children:
            addnode(k)
            gv.append('"%s" -> "%s";' % (n.label, k.label))

    addnode(zt)
    gv.append("}")
    return "\n".join(gv)


def ft_nr2pt(ftt, names):
    nodes = {}
    for i, n in enumerate(names):
        nodes[i] = ZSNode(n)
    t = buildtree(ftt[0], len(names), ftt[1])
    for i, n in enumerate(t):
        nid = -i - 1
        if not nid in nodes:
            nodes[nid] = ZSNode('c')
        if n[1] != None:
            s = [n[1], n[2]]
        else:
            s = list(n[0])
        for v in s:
            if not v in nodes:
                nodes[v] = ZSNode('c')
            nodes[nid].addkid(nodes[v])
    return nodes[-1]


def anytree2dot(t, names=None):
    ##	This implementation is prefered, but currently loses reliability measures
    ##	from ctrees
    ##	if type(t) != ZSNode:
    ##		t = t2zst(t, names)
    ##	return zst2dot(t)
    if type(t) in [str, unicode]:
        return t
    elif type(t) in [tuple, list]:
        return ctree2dot(t[0], names, t[1])
    elif type(t) == Tree:
        return tree2dot(t, names)
    elif type(t) == ZSNode:
        return zst2dot(t)
    else:
        raise ValueError("%s isn't a know type of Tree" % (type(t),))


def t2zst(t, names):
    """
    distributer function that checks the type of tree structure t and calls the
    appropriate *2zst function to produce a Zhang-shasha compatible tree.

    """
    if type(t) == Tree:
        return pct2zst(t, names)
    else:
        return ft_nr2pt(t, names)


def zssdist(t1, t2, names):
    """
    t1: Tree, t2:Tree -> x

    Return the edit distance, as computed by zss.compare, between t2zst-compatible trees t1, t2
    """
    t1, t2 = map(lambda x: t2zst(x, names), [t1, t2])
    return zscmp.distance(t1, t2)