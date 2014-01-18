#!/usr/bin/env python
# encoding: utf-8
#Created by  on 2010-12-07.

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

import numpy as np
from gicdat.control import report
from ll import ece, ucse, DRMODES, eqrc, lltest, LENGTH, LEAD, TESTPROP, BOOTSTRAP
import numpy as np
from gicdat.base import Transform
import gicdat.doc as gd
from acell import acell


def cSpaceSet(d, cmode='istac', clevel=3):
    #FIXME: assumes particular structure for D. Should be a transform.
    uc1 = ucse(d['s1'], 10000, LENGTH)
    uc2 = ucse(d['s2'], 10000, LENGTH)
    rc1 = ece(d['s1'], d['evts1'], LENGTH, LEAD)
    rc2 = ece(d['s2'], d['evts2'], LENGTH, LEAD)
    istacs = {}
    istacs['uc'] = comp(uc2, uc1, clevel)
    istacs['rc'] = comp(rc2, rc1, clevel)
    istacs['c1'] = comp(rc1, uc1, clevel)
    istacs['c2'] = comp(rc2, uc2, clevel)
    istacs['j'] = comp(eqrc((rc1, rc2)), np.column_stack([uc1, uc2]), clevel)
    return istacs


def comp2s(stim1, evts1, stim2, evts2, length=LENGTH, lead=LEAD, compress='no', clevel=0, testprop=TESTPROP,
           bootstrap=BOOTSTRAP, report=None):
    '''
    Like compare, but considers the case where there are two stimuli in
    addition to two response sets
    '''
    rc1 = ece(stim1, evts1, length, lead)
    rc2 = ece(stim2, evts2, length, lead)
    if compress in DRMODES:
        uc = np.column_stack([ucse(stim1, 10000, length),
                              ucse(stim2, 10000, length)])
        ce = eqrc((rc1, rc2))
        cspace = DRMODES[compress](ce, uc, clevel)
        if report:
            report('Using %i components' % cspace.shape[0])
        rc1 = np.dot(cspace, rc1)
        rc2 = np.dot(cspace, rc2)
    return lltest(rc1, rc2, testprop, bootstrap)


class Compare2s(Transform):
    sig = {}
    defaults = {'length': 60, 'lead': 40, 'compress': 'no', 'clevel': 0.0, 'testprop': .2, 'bootstrap': 10,
                'report': False, 'outpath': 'fitness'}

    def run(self, doc, pars, out, messages):
        paths = dict([(n, doc[pars[n]]) for n in ['stim1', 'stim2', 'evts1', 'evts2']])
        ll = self.callWith(comp2s, paths, pars)
        c12 = (ll[0, 0, 0] - ll[0, 1, 0]) / (ll[0, 0, 1] + ll[0, 1, 1])
        c21 = (ll[1, 1, 0] - ll[1, 0, 0]) / (ll[1, 0, 1] + ll[1, 1, 1])
        out[pars['outpath']] = min(c12, c21)
        out[pars['outpath'] + '.raw'] = ll


compare2 = Compare2s()

if __name__ == "__main__":
    import gwn

    s = gwn.getstim().fuse(gwn.getstim(band=(1, 400)))
    s['m1.outpath'] = 'evts1'
    s['m1.stimpath'] = 'stim5_150'
    d, r = acell(s, 'm1')
    for st in r:
        print(st)
    s['m2.outpath'] = 'evts2'
    s['m2.stimpath'] = 'stim1_400'
    d2, r = acell(s, 'm2')
    for st in r:
        print(st)
    d = d.fuse(d2)
    d['stim1'] = s['stim5_150']
    d['stim2'] = s['stim1_400']
    cmo = {'evts1': 'evts1', 'evts2': 'evts2', 'stim1': 'stim1', 'stim2': 'stim2'}
    print d['evts1']
    d3, r = compare2(d, cmo)
    print(d3['fitness'], d3.d.fitness.raw)

