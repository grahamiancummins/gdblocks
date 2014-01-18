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

import timingUI as ti
import selectUI as si
import tests


def allstims(cdoc):
    stims = set()
    for k in cdoc.keys(0, 'cell'):
        d = cdoc[k]['stimclasses']
        l = [d[s]['file'] for s in d]
        stims = stims.union(l)
    return stims


def stimsfor(cdoc, cell):
    c = cdoc[cell]
    gc = set(c['cond1.stims'])
    gc = gc.intersection(set(c['cond2.stims']))
    l = [c['stimclasses.stim%i.file' % k] for k in gc]
    return sorted(l)


def hasstims(cdoc, stims):
    stims = set(stims)
    cells = []
    for k in cdoc.keys(0, 'cell'):
        cst = set(stimsfor(cdoc, k))
        if stims.issubset(cst):
            cells.append(k)
    return cells


def jitt():
    d = tests.p_cdoc()
    c = d['condtgc']
    cj1 = ti.tin.jitter(c, 0, 0, 1)
    cj2 = ti.tin.jitter(c, 0, 0, 2)


def condmi(c, jit=0, jitsd=0, dmeth='vdps', q=1000,
           nclust=50, mim='direct', debias=None):
    if jit:
        c = ti.tin.jitter(c, jitsd, 0, jit)
    else:
        c = c.copy()
    dm = ti.tin.dist.dist(c['evts'], dmeth, q)
    ec = ti.tin.clustdm('tree', dm, nclust)
    c['raw'] = c['evts']
    c['dm'] = dm
    c['evts'] = ec
    if debias:
        mi, ie, oe = ti.tin.minf_db(c, mim, debias)
    else:
        mi, ie, oe = ti.tin.minf(c, mim, True)
        mi = [mi]
    return (mi, ie, oe, c)

