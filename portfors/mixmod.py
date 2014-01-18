#!/usr/bin/env python
# encoding: utf-8
#Created by  on 2009-06-17.

# Copyright (C) 2009 Graham I Cummins
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

import numpy as np
from numpy import linalg
from numpy.random import randn
from tempfile import mkdtemp
import os
import gicdat.doc as gd
from gicdat.enc import astuple, flat


MMCMD = os.path.join(os.environ['HOME'], 'bin/mixmod')


def _parsemm(l):
    mods = []
    lsingle = None
    cmod = {}
    for line in l:
        line = line.strip()
        if not line:
            continue
        line = map(float, line.split(" "))
        if len(line) == 1:
            if lsingle:
                if cmod:
                    mods.append(cmod)
                cmod = {'bic': lsingle, 'components': []}
                mods.append(cmod)
            lsingle = line[0]
        elif lsingle:
            ccomp = {'prop': lsingle, 'mean': line, 'cov': []}
            cmod['components'].append(ccomp)
            lsingle = None
        else:
            ccomp['cov'].append(line)
    return mods


def _parsemm1d(l):
    mods = []
    cmod = {}
    tdens = 0
    ccomp = []
    for line in l:
        line = line.strip()
        if not line:
            continue
        line = map(float, line.split(" "))
        if len(line) != 1:
            raise IOError("parsemm1d called an multi-d data")
        v = line[0]
        if not cmod:
            cmod = {'bic': v, 'components': []}
            tdens = 0
            ccomp = []
            continue
        ccomp.append(v)
        if len(ccomp) >= 3:
            tdens += ccomp[0]
            cmod['components'].append({'prop': ccomp[0], 'mean': [ccomp[1]],
                                       'cov': [[ccomp[2]]]})
            ccomp = []
        if tdens >= .999:
            mods.append(cmod)
            cmod = {}
    return mods


def _writeMixControl(dn, data, clusters, model, weighted=False):
    curdir = os.getcwd()
    os.chdir(dn)
    if weighted:
        wt = data[:, -1]
        if min(wt) <= 0.0:
            print("WARNING: Weight vector must be strictly positive. Switching weighted mode off")
            weighted = False
        else:
            wt = wt / wt.min()
            wt = wt.round().astype(int64)
            data = data[:, :-1]
    cf = open('control.xem', 'w')
    cf.write("NbLines\n\t%i\n" % (data.shape[0],))
    cf.write("PbDimension\n\t%i\n" % (data.shape[1],))
    cf.write("NbNbCluster\n\t%i\n" % (len(clusters),))
    cf.write("ListNbCluster\n\t%s\n" % (" ".join(map(str, clusters)),))
    cf.write("NbModel\n\t1\n")
    cf.write("ListModel\n\t%s\n" % model)
    cf.write("DataFile\n\tdata.dat\n")
    if weighted:
        cf.write("WeightFile\n\twt.wgt\n")
    cf.close()
    df = open('data.dat', 'w')
    for i in range(data.shape[0]):
        l = " ".join(map(str, list(data[i, :])))
        df.write(l + "\n")
    df.close()
    if weighted:
        wf = open('wt.wgt', 'w')
        for n in wt:
            wf.write(str(n) + "\n")
    os.chdir(curdir)


def _read_partition(fn):
    z = np.array([map(int, l.split()) for l in open(fn).readlines() if l.strip()])
    return np.nonzero(z)[1]


def mmcall(data, clusters, model="Gaussian_pk_Lk_Ck", weighted=False, reps=1):
    dn = mkdtemp()
    _writeMixControl(dn, data, clusters, model, weighted)
    curdir = os.getcwd()

    os.chdir(dn)
    #print dn
    bbic = -np.inf
    bpart = []
    for i in range(reps):
        try:
            os.system("%s control.xem" % MMCMD)
            out = open('numericComplete.txt').readlines()
            if data.shape[1] == 1:
                mod = _parsemm1d(out)
            else:
                mod = _parsemm(out)
        except:
            os.chdir(curdir)
            os.system('rm -rf %s' % dn)
            _writeMixControl(curdir, data, clusters, model, weighted)
            print('mixmod failed. Wrote control.xem and data.dat in the current directory')
            raise
        #open('test.txt', 'w').write("".join(out))
        best = 0
        bic = mod[0]['bic']
        for i in range(1, len(mod)):
            if mod[i]['bic'] < bic:
                best = i
                bic = mod[i]['bic']
        if bic > bbic:
            bmod = mod[best]
            bbic = bic
            bpart = _read_partition("BICpartition.txt")
    os.chdir(curdir)
    os.system('rm -rf %s' % dn)
    d = _asdoc(bmod)
    d['partition'] = tuple(bpart)
    return d


def _asdoc(mod):
    d = gd.Doc()
    d['bic'] = mod['bic']
    props = []
    components = []
    for i, c in enumerate(mod['components']):
        n = "c%i" % i
        props.append(c['prop'])
        components.append(n)
        d[n] = {'mean': astuple(c['mean']), 'cov': astuple(c['cov'])}
    d['proportions'] = tuple(props)
    d['components'] = tuple(components)
    return d


def _test(shape=(77, 3), clusters=(1, 2, 3, 4, 5), model="Gaussian_pk_Lk_Ck", wt=False):
    dat = apply(randn, shape)
    eoo = np.arange(0, shape[0], 2)
    dat[eoo] += 3
    if wt:
        dat[:, -1] = dat[:, -1] - dat[:, -1].min() + .1
    return mmcall(dat, clusters, model, wt, 5)


def _wlcf():
    dat = apply(randn, (77, 3))
    eoo = np.arange(0, 77, 2)
    dat[eoo] += 3
    _writeMixControl(os.getcwd(), dat, [1, 2, 3], "Gaussian_pk_Lk_Ck", False)


def evaluate(mm, pts):
    pts = np.array(pts)
    if len(mm['components']) == 0:
        return np.ones(pts.shape[0]) / mm['support']
    nd = len(mm[mm['components'][0]]['mean'])
    res = np.zeros((len(pts), len(mm['components']) ))
    for j, c in enumerate(mm['components']):
        w = mm['proportions'][j]
        mea = np.array(mm[c]['mean'])
        if nd == 1:
            sig = mm[c]['cov'][0][0]
            norm = 1.0 / np.sqrt(2 * np.pi * sig)
            p = pts.flatten() - mea[0]
            e = (p ** 2) / (2 * sig)
            res[:, j] = w * norm * np.exp(-1 * e)
        else:
            norm = 1.0 / ( np.sqrt(linalg.det(mm[c]['cov'])) * (2 * np.pi) ** (nd / 2.0))
            icov = linalg.inv(mm[c]['cov'])
            for i, p in enumerate(pts):
                x = p - mea
                e = -.5 * np.dot(np.dot(x, icov), x)
                res[i, j] = w * norm * np.exp(e)
    return res.sum(1)