#!/usr/bin/env python -3
# encoding: utf-8

#Created by gic on 
#Wed Nov 24 16:51:14 CST 2010

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

import gicdat.doc as gd
import numpy as np
import acell, gwn
from gicdat.stdblocks.tags import timeseries_t

from gdblocks.optim.ga import Optimizer, Model, ParamRange, Store, GA_Prop


def _parse(pars, nd):
    o = [1]
    for i in range(nd - 1):
        o.append(o[-1] + int(pars[i]))
    o = tuple(o)
    m = pars[nd - 1:2 * nd - 1]
    s = pars[2 * nd - 1:]
    return (o, m, s)


def npars(nd):
    n = nd - 1 + nd + (nd ** 2 + nd) / 2
    return n


def parse(pars, nd):
    nppm = npars(nd)
    if len(pars) == nppm:
        m1 = ((1,), (1,), (.05,))
    else:
        m1 = _parse(pars[:nppm], nd)
        pars = pars[nppm:]
    m2 = _parse(pars, nd)
    return (m1, m2)


class AcellCross(Model):
    def __init__(self, pars):
        self.compress = pars['compress']
        self.clevel = pars['clevel']
        self.doc = gwn.getstim('doc', pars['slen'])
        self.snn = self.doc.find(timeseries_t, True)
        self.ac = acell.acell()
        self.cm = acell.compare()
        self.nd = pars['nd']
        Model.__init__(self, pars)

    def eval(self, pars):
        (o1, m1, s1), (o2, m2, s2) = parse(pars, self.nd)
        try:
            self.ac(self.doc, stimulus=self.snn, off=o1, mu=m1, sigma=s1, outpath='evt1', cid=0)
            self.ac(self.doc, stimulus=self.snn, off=o2, mu=m2, sigma=s2, outpath='evt2', cid=1)
            self.cm(self.doc,
                    **{'evts1': 'evt1', 'bootstrap': 20, 'evts2': 'evt2', 'stim': self.snn, 'outpath': self.snn})
            f = -1 * self.doc['stimulus']['fitness']
            if f <= 1:
                return (f, ())
            else:
                self.cm(self.doc, **{'compress': self.compress, 'clevel': self.clevel, 'evts1': 'evt1', 'evts2': 'evt2',
                                     'bootstrap': 20, 'stim': self.snn, 'outpath': self.snn})
                f2 = self.doc['stimulus']['fitness']
                if f2 < -1:
                    f2 = min(1.0, f + f2)
                    return (f2, ())
                else:
                    return (f + f2, ())
        except np.linalg.linalg.LinAlgError:
            return (-100, ())


def run(fn, pfile):
    pars = eval(open(pfile).read())
    ofr = np.array(pars['ofr'])
    mrf = np.array(pars['mrf'])
    mr = np.array(pars['mr'])
    sr = np.array(pars['sr'])
    srd = np.array(pars['srd'])
    ran = np.row_stack([ofr, ofr, mr, mr, mr, srd, sr, sr, srd, sr, srd])
    ran = np.row_stack([ran, ran])
    p = ParamRange(ran)
    s = Store(fn + '.giclog')
    model = AcellCross(pars['mod'])
    alg = GA_Prop(pars['ga'], p)
    o = Optimizer(pars['base'], model, alg, s)
    o.run()

#return(p, par)

if __name__ == '__main__':
    import sys

    fn = sys.argv[1]
    pars = sys.argv[2]
    run(fn, pars)
