#!/usr/bin/env python -3
# encoding: utf-8

#Created by gic on Thu Jan 20 11:44:48 CST 2011

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

from gicdat.control import report
from gicdat.enc import flat, trange, tshape, flatiter
from vis import RasterGrid
from shutil import copy as fcopy
import gicdat.doc as gd
import gicdat.io as gio
import matplotlib.pyplot as plt
import numpy as np
import os
import re

BASEDIR = os.path.join(os.environ['HOME'], 'project/christine')


def _numOrRan(s):
    m = re.search("(\d+)-?(\d+)?", s)
    start, stop = m.groups()
    if not stop:
        return [int(start)]
    else:
        return range(int(start), int(stop) + 1)


def _hasbic(s):
    if s.lower().startswith('bic'):
        return "drug"
    else:
        return "control"


def stash(dname, subdir="bicIC"):
    '''
    Copy all the PST files found in directory "dname" to the indicated
    subdirectory of the project directory. This function creates container
    directories with the same name as the files (minus extension)

    '''
    for fn in os.listdir(dname):
        if fn.endswith('.pst'):
            ffn = os.path.join(dname, fn)
            dn = os.path.join(BASEDIR, subdir, os.path.splitext(fn)[0])
            if not os.path.isdir(dn):
                os.mkdir(dn)
            fcopy(ffn, dn)


def _descstim(d):
    if d['type'] == 'stored file':
        return d['file'].split('.')[0]
    elif d['type'] == 'tone':
        return "tone %i" % (int(d.get('frequency')),)
    else:
        return d.get('value')


def _shift(evtl, v):
    try:
        v = float(v)
    except:
        return evtl
    nel = []
    for et in evtl:
        nel.append(tuple(np.array(et) - v))
    return tuple(nel)


def traces(d, ns, st, shift=True):
    tr = {}
    for tk in d[ns]:
        if tk.startswith('trace'):
            trn = int(tk[5:])
            tra = d[ns][tk]
            stim = tra[st] or {'type': 'No Stimulus', 'value': 'No Stimulus', 'delay': 0.0, 'attenuation': 1000.0,
                               'duration': 0}
            trd = {}
            desc = _descstim(stim)
            if shift:
                spks = _shift(tra['events'], stim['delay'] * 1000)
            else:
                spks = tra['events']
                trd['stim_delay'] = stim['delay']
            trd['evts'] = spks
            trd['stim_type'] = desc
            trd['stim_atten'] = stim['attenuation']
            trd['stim_dur'] = stim['duration']
            tr[trn] = trd
    return tr


def getcell(cellid, dname='bicIC', sheet='fulldatasheet.ods', tab='sheet1.table', cellidlab='cell',
            mouseid='data', testn='test', cond='condition', pstdirp='Mouse ', redundantdirs=True, condgroup=int,
            stimchan='stim.ch0', shift=True):
    '''
    Return a data document for a single cell. Parameters are the same as for condition, except for cellid, which
    is an integer indicating which cell to get. As long as "fulldatasheet.ods" is present in the named "dname", and
    lists all available cells in Zach/Graham format, then no parameters other than cellid and dname should need to be
    changed.
    '''
    dname = os.path.join(BASEDIR, dname)
    ds = gio.read(os.path.join(dname, sheet))[tab]
    l = list(ds[0, :])
    try:
        cellcol = [i for (i, s) in enumerate(l) if s.lower().startswith(cellidlab)][0]
        mousecol = [i for (i, s) in enumerate(l) if s.lower().startswith(mouseid)][0]
        testcol = [i for (i, s) in enumerate(l) if s.lower().startswith(testn)][0]
        condcol = [i for (i, s) in enumerate(l) if s.lower().startswith(cond)][0]
    except IndexError:
        raise KeyError('Data sheet doesnt contain the specified columns')
    clines = [(ds[i, mousecol], ds[i, testcol], ds[i, condcol]) for i in range(ds.shape[0]) if
              ds[i, cellcol] == str(cellid)]
    if not clines:
        raise KeyError('No tests in this data sheet for this cell')
    if not all([clines[i][0] == clines[0][0] for i in range(1, len(clines))]):
        report('WARNING: single cell reported in multiple recording tracks !!??')
        sd = gd.Doc()
        for mn in set([clines[i][0] for i in range(len(clines))]):
            pstn = pstdirp + mn + '.pst'
            if redundantdirs:
                pstn = os.path.join(pstdirp + mn, pstn)
            pstn = os.path.join(dname, pstn)
            sd = sd.fuse(gio.read(pstn))
    else:
        pstn = pstdirp + clines[0][0] + '.pst'
        if redundantdirs:
            pstn = os.path.join(pstdirp + clines[0][0], pstn)
        pstn = os.path.join(dname, pstn)
        sd = gio.read(pstn)
    d = gd.Doc()
    for l in clines:
        tests = _numOrRan(l[1])
        for it in tests:
            otn = 'test%i' % it
            trd = traces(sd, otn, stimchan, shift)
            for trn in trd:
                tr = trd[trn]
                tr['condition'] = condgroup(l[2])
                tr['cell'] = cellid
                nn = "te%itr%i" % (it, trn)
                d.set(nn, tr)
    return d


def condition(dname='bicIC', sheet='datasheet.ods', tab='sheet1.table', cellid='cell', mouseid='data',
              testn='test', cond='condition', pstdirp='Mouse ', redundantdirs=True, writefile='conditioned.gic',
              condgroup=int, stimchan='stim.ch0'):
    '''
    Return a data document containing every test specified in the data sheet
    "sheet". "dname" indicates a directory (subdirectory of BASEDIR) to search for
    the data sheet and data files (or "mouse" directories).

    tab, cellid, mouseid, testn, and cond specify the layout of the sheet (which
    table is used and what the column labels for the relevant metadata are. These
    can be left as default values if using the sort of sheet Zach Mayko and I
    (Graham Cummins) have typically been using. Pstdirp and redundantdirs specify
    the layout of the data directory structure, and again can remain at default if
    your structure looks like Christine Portfor's typical layout. Writefile
    specifies a file name to store the new document in (or omit it to not save the
    doc).

    condgroup is a function that is applied to the value of the "condition" metadata
    The default (cast to int) is OK if condition labels are integers, and all distinctly
    labeled conditions should be treated differently (otherwise, use a function that
    makes similarity classes in some way, using the raw values of "condition").

    Stimchan specifies where the relevant stimulus data is stored by Batlab.
    This code assumes a single channel stimulus on channel 0. Although this
    parameter can be changed to deal with single channels on some other
    channel, any significant change to the stimulus will probably break
    most code in this module.

    '''
    dname = os.path.join(BASEDIR, dname)
    ds = gio.read(os.path.join(dname, sheet))[tab]
    l = list(ds[0, :])
    try:
        cellcol = [i for (i, s) in enumerate(l) if s.lower().startswith(cellid)][0]
        mousecol = [i for (i, s) in enumerate(l) if s.lower().startswith(mouseid)][0]
        testcol = [i for (i, s) in enumerate(l) if s.lower().startswith(testn)][0]
        condcol = [i for (i, s) in enumerate(l) if s.lower().startswith(cond)][0]
    except IndexError:
        raise KeyError('Data sheet doesnt contain the specified columns')
    d = gd.Doc()
    dfiles = {}
    for i in range(1, ds.shape[0]):
        l = ds[i, :]
        try:
            cid = int(l[cellcol])
            mid = l[mousecol]
            cond = l[condcol]
            if condgroup:
                cond = condgroup(cond)
            tests = _numOrRan(l[testcol])
        except:
            report('failed to parse data sheet line %i' % i)
            continue
        if not mid in dfiles:
            pstn = pstdirp + mid + '.pst'
            if redundantdirs:
                pstn = os.path.join(pstdirp + mid, pstn)
            pstn = os.path.join(dname, pstn)
            try:
                dfiles[mid] = gio.read(pstn)
                report('loading %s' % pstn)
            except:
                report('failed to open data file for %s (tried path %s)' % (mid, pstn))
                dfiles[mid] = 'BAD'
        if dfiles[mid] == 'BAD':
            continue
        metas = {'animal': mid, 'condition': cond, 'cell': cid}
        for it in tests:
            otn = 'test%i' % it
            try:
                trd = traces(dfiles[mid], otn, stimchan)
            except:
                report("Cant find a test %s for penetration %s" % (otn, mid))
                continue
            for trn in trd:
                tr = trd[trn]
                tr.update(metas)
                nn = "r%s_te%itr%i" % (mid, it, trn)
                d.set(nn, tr)
    if writefile:
        gio.write(d, os.path.join(dname, writefile))
    return d


def _tupstr(t):
    s = []
    for v in sorted(t):
        if type(v) == float:
            v = "%.4g" % v
        s.append(str(v))
    s = "[" + ', '.join(s) + ']'
    return s


def _rpp(d):
    npres = 0
    nspks = 0
    for k in d:
        if d[k]['evts']:
            npres += len(d[k]['evts'])
            nspks += len(flat(d[k]['evts']))
    if npres == 0:
        return -1
    return float(nspks) / float(npres)


def describe(d, detail=True):
    ntr = 0
    attr = {}
    mipr = np.inf
    mapr = -np.inf
    misp = np.inf
    masp = -np.inf
    fstsp = np.inf
    lstsp = -np.inf
    for k in d.keys():
        tr = d[k]
        if not type(tr) == gd.Doc:
            continue
        ntr += 1
        for sk in tr:
            if sk == 'evts':
                ts = tshape(tr[sk])[0]
                ran = trange(tr[sk])
                fstsp = min(fstsp, ran[0])
                lstsp = max(lstsp, ran[1])
                mipr = min(mipr, ts[0])
                mapr = max(mapr, ts[0])
                if type(ts[1]) == tuple:
                    misp = min(misp, ts[1][0])
                    masp = max(masp, ts[1][1])
                else:
                    misp = min(misp, ts[1])
                    masp = max(masp, ts[1])
            else:
                if not sk in attr:
                    attr[sk] = set()
                attr[sk].add(tr[sk])
    s = ["%i traces" % (ntr,)]
    s.append("%i to %i presentations giving %i to %i spikes on [%i, %i]" % (mipr, mapr, misp, masp, fstsp, lstsp))
    s.append('Attributes:---')
    for k in sorted(attr):
        s.append('%s in %s' % (k, _tupstr(attr[k])))
        s.append('--------')
    if detail:
        for cond in allv(d, 'condition'):
            for sa in allv(d, 'stim_atten'):
                pres = cases(d, condition=[cond], stim_atten=[sa])
                s.append('condition %i, attenuation %.3g : %.2g spks/presentation' % (cond, sa, _rpp(pres)))
    print '\n'.join(s)


def allv(d, cond):
    kk = [k for k in d.iter() if k.split('.')[-1] == cond]
    return sorted(set([d[k] for k in kk]))


def stimtones(d):
    stims = allv(d, 'stim_type')
    tones = []
    calls = []
    for s in stims:
        if s.startswith('tone'):
            tones.append(s)
        else:
            calls.append(s)
    calls = sorted(calls)
    if 'No Stimulus' in calls:
        calls.remove('No Stimulus')
        calls.insert(0, 'No Stimulus')
    return (calls, sorted(tones, gd._cmptint))


def find(d, **kwargs):
    keys = None
    if 'keys' in kwargs:
        keys = kwargs['keys']
    z = {}
    for x in kwargs:
        if x == 'keys':
            continue
        if type(kwargs[x]) in (str, unicode):
            z[x] = "=" + kwargs[x]
        else:
            z[x] = kwargs[x]
    return [k for k in d.find(z, keys=keys)]


def cascadesort(x, y, d, sortord):
    x = d[x]
    y = d[y]
    for s in sortord:
        cv = cmp(x[s], y[s])
        if cv:
            return cv
    return 0


def csorted(d, keys, sortord=('stim_type', 'stim_atten')):
    return sorted(keys, lambda x, y: cascadesort(x, y, d, sortord))


def fuse(d, keys, sortord=('stim_type', 'stim_atten')):
    keys = csorted(d, keys, sortord)
    l = [d[k]['evts'] for k in keys]
    c = ()
    for t in l:
        c = c + t
    return c


def reduce(d, keep=('condition', 'cell', 'stim_type', 'stim_atten'), require=(), writefile='bicIC/reduced.gic'):
    #require e.g. (('stim_atten', 0.0),)
    d2 = gd.Doc()
    casenames = {}
    cname = 0
    for k in d.keys():
        if not type(d[k]) == gd.Doc:
            continue
        skip = False
        for c in require:
            if not c[1] == d[k + '.' + c[0]]:
                skip = True
                #print("%s fails condr=RasterGrid(1, len(rgrid[0]))itions that %s = %s (it is %s)" % (k, c[0], str(c[1]), str(d[k+'.'+c[0]])))
                break
        if skip:
            continue
        s = tuple([d[k + '.' + ke] for ke in keep])
        if s in casenames:
            nn = casenames[s]
        else:
            nn = 't%i' % cname
            cname += 1
            casenames[s] = nn
            for i in range(len(s)):
                d2.set(nn + '.' + keep[i], s[i])
        nn = nn + '.evts'
        if d2[nn]:
            d2[nn] = d2[nn] + d[k + '.evts']
        else:
            d2[nn] = d[k + '.evts']
    if writefile:
        gio.write(d, os.path.join(BASEDIR, writefile))
    return d2


def cases(d, **kwargs):
    '''
    return a subset of d as a new document. kwargs can contain tag=list pairs.
    For each one, a trace appears in the new document only if it has a value of
    this tag which appears in this list
    '''
    d2 = gd.Doc()
    for k in d:
        if not type(d[k]) == gd.Doc:
            continue
        sd = d[k]
        for kk in kwargs:
            v = kwargs[kk]
            if not type(v) in [tuple, list, np.ndarray]:
                v = [v]
            if not sd[kk] in v:
                break
        else:
            d2.set(k, sd, safe=False)
    return d2


def _win(t, mi, ma):
    new = []
    for es in t:
        new.append(tuple([e for e in es if e >= mi and e <= ma]))
    return tuple(new)


def stimwin(d, laten=0, lag=50000, mode='local'):
    '''
    return a patch containing events in d only between indexes laten and stim_dur+lag
    mode may be local -> stim_dur is the stim_dur for the current group
                max-> stim_dur is the max stim_dur in the document
                min -> stim_dur is the min stim_dur in the document
                mean -> stim_dur is the average stim_dur in the document
    '''
    patch = gd.Doc()
    keys = d.findall({'evts': '_', 'stim_dur': "X::X>0"})
    if mode == 'local':
        stim_dur = None
    else:
        durs = np.array([d[k + '.stim_dur'] for k in keys])
        stim_dur = eval('durs.%s()' % mode)
        patch['windur'] = stim_dur * 1000
    for k in keys:
        evts = d[k + '.evts']
        if stim_dur == None:
            dur = d[k + '.stim_dur']
        else:
            dur = stim_dur
        if d[k + '.stim_delay']:
            evts = _shift(evts, 1000 * d[k + 'stim_delay'])
        dur = dur * 1000 # batlab stores duration in ms, but spike times are in micro-s
        patch[k + '.evts'] = _win(evts, laten, dur + lag)
    return patch


def raster(sd):
    rgrid = [[sd[k] for k in sd if k.startswith('cond')]]
    r = RasterGrid(1, len(rgrid[0]))
    r.clear()
    r.add(rgrid)
    r.draw()


def histpl(d, k, bw=4.0, rng=(0, 200.0)):
    evts = d[k + '.evts']

    evts = np.array(flat(evts))
    evts = evts / 1000.0
    nb = (rng[1] - rng[0]) / bw

    plt.hist(evts, nb, rng)


def setrasters(d, vsplit='cell', hsplit='condition', csplit='stim_type',
               sort=('stim_atten',), fig=1, save=None):
    vkeys = allv(d, vsplit)
    hkeys = allv(d, hsplit)
    ckeys = allv(d, csplit)
    rg = RasterGrid(len(vkeys), len(hkeys), fig)
    rg.clear()
    for i in range(len(vkeys)):
        for j in range(len(hkeys)):
            rg.titles[i][j] = "%s,%s" % (vkeys[i], hkeys[j])
    for c, cv in enumerate(ckeys):
        grid = []
        keys_c = find(d, **{csplit: cv})
        for vk in vkeys:
            grid.append([])
            keys_v = find(d, **{vsplit: vk, 'keys': keys_c})
            for hk in hkeys:
                keys = find(d, **{hsplit: hk, 'keys': keys_v})
                dat = fuse(d, keys, sort)
                grid[-1].append(dat)
        if keys_c:
            sdur = d[keys_c[0] + '.stim_dur']
            sstrt = d[keys_c[0] + '.stim_delay'] or 0.0
        else:
            sdur = None
        if sdur:
            rg.add(grid, c, {'alpha': .9, 'roi': [(sstrt, sstrt + sdur)]})
        else:
            rg.add(grid, c)
    rg.draw()
    if save:
        plt.savefig(save)


def cellrasters(d, cell=1):
    setrasters(d.subset(d.find({'cell': cell})))


def manyrasters(d, maxrows=4, maxcols=3, vsplit='condition', hsplit='cell', csplit='stim_type', sort=('stim_atten',),
                lab='calls'):
    f = plt.figure(1)
    f.set_figwidth(10)
    f.set_figheight(8)
    vkeys = allv(d, vsplit)
    vkeyg = []
    while vkeys:
        v, vkeys = vkeys[:maxrows], vkeys[maxrows:]
        vkeyg.append(v)
    hkeys = allv(d, hsplit)
    hkeyg = []
    while hkeys:
        h, hkeys = hkeys[:maxcols], hkeys[maxcols:]
        hkeyg.append(h)
    for vg in vkeyg:
        for hg in hkeyg:
            print(vg, hg)
            sd = cases(d, **{vsplit: vg, hsplit: hg})
            if len(vkeyg) > 1 and len(hkeyg) > 1:
                fname = "%s%s%s%s%s.png" % (vsplit, str(vg[0]), hsplit, str(hg[0]), lab)
            elif len(vkeyg) == 1:
                fname = "%s%s_%s_%s.png" % (hsplit, str(hg[0]), str(hg[-1]), lab)
            elif len(hkeyg) == 1:
                fname = "%s%s_%s_%s.png" % (vsplit, str(vg[0]), str(vg[-1]), lab)
            setrasters(sd, vsplit, hsplit, csplit, sort, 1, save=fname)
            plt.close()


def rastersbycell(d, lab=''):
    f = plt.figure(1)
    f.set_figwidth(10)
    f.set_figheight(8)
    cells = allv(d, 'cell')
    for c in cells:
        sd = cases(d, cell=[c])
        fname = "cell%i%s.png" % (c, lab)
        setrasters(sd, save=fname)


def tunecurve(d, cond):
    c = cases(d, stim_type=stimtones(d)[1], condition=[cond])
    tones = allv(c, 'stim_type')
    tints = [int(t.split()[1]) for t in tones]
    sorin = np.argsort(tints)
    attens = sorted(allv(c, 'stim_atten'))
    out = np.zeros((len(tints), len(attens)))
    for i in range(len(tints)):
        for j in range(len(attens)):
            t = fuse(c, find(c, stim_type=tones[sorin[i]], stim_atten=attens[j]), ())
            nspks = len([x for x in flatiter(t)])
            out[i, j] = float(nspks) / len(t)

    f = plt.figure(1)
    plt.clf()
    extent = [min(tints), max(tints), max(attens), min(attens)]
    aspect = .8 * (extent[1] - extent[0]) / (extent[2] - extent[3])
    plt.imshow(out.transpose(), aspect=aspect, extent=extent, interpolation='nearest')
    plt.colorbar()
    return out


def plotTC(d, tc):
    c = cases(d, stim_type=stimtones(d)[1])
    tones = allv(d, 'stim_type')
    tints = sorted([int(t.split()[1]) for t in tones])
    attens = sorted(allv(c, 'stim_atten'))


if __name__ == '__main__':
    r = gio.read('reduced.gic')
    calls, tones = stimtones(r)
    rc = cases(r, stim_type=calls)
    rastersbycell(rc, 'calls')
#	rt = cases(r, stim_type=tones)
#	print('calls')
#	manyrasters(rc)
#	print('tones')
#	manyrasters(rt, lab='tones')



