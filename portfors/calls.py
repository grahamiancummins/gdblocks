#!/usr/bin/env python -3
# encoding: utf-8

#Created by Graham Cummins on 

# Copyright (C) 2011 Graham I Cummins
# This program is free software; you can redistribute it and/or modify it
#underthe terms of the GNU General Public License as published by the Free
#Software Foundation; either version 2 of the License, or (at your option) 
#any later version.

#This program is distributed in the hope that it will be useful, but WITHOUT 
#ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
#FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
#for more details.

#You should have received a copy of the GNU General Public License along with
#this program; if not, write to the Free Software Foundation, Inc., 59 Temple
#Place, Suite 330, Boston, MA 02111-1307 USA


from __future__ import print_function, unicode_literals
import numpy as np
import wave
import gicdat.doc as gd
from gicdat.base import Parser


def read(f, asfloat=True):
    a = np.reshape(np.fromstring(open(f).read(), np.int16), (-1, 1))
    fs = 400000.0
    if asfloat:
        a = a.astype(np.float64) / 32767
    return fs, a


def readwav(f):
    wf = wave.open(f)
    head = {}
    fs = wf.getframerate()
    nf = wf.getnframes()
    nc = wf.getnchannels()
    nbits = wf.getsampwidth()
    data = wf.readframes(nf)
    dt = np.dtype("i%i" % (nbits,))
    data = np.fromstring(data, dt)
    data = np.reshape(data, (-1, nc))
    data = data.astype('f4') / 2 ** (8 * nbits - 1)
    return (fs, data)


def writewav(f, dat, fs):
    r = np.abs(dat).max()
    if r > 1.0:
        dat = dat / r
    dat = dat * 32767
    dat = dat.astype('i2')
    wf = wave.open(f, 'w')
    wf.setnchannels(dat.shape[1])
    wf.setsampwidth(2)
    wf.setframerate(fs)
    wf.writeframes(dat.tostring())


def towav(fn, wfn, fs=44100, norm=True):
    _, a = read(fn, True)
    if norm:
        a = a / np.abs(a).max()
    writewav(wfn, a, fs)


class CallParse(Parser):
    '''
    This parser handles reading (only) of batlab call1 files

    '''

    canread = ('application/batlab.call1',)
    #canwrite = ('application/batlab.call1',)
    extensions = {'call1': 'application/batlab.call1'}

    def read(self, stream, filetype, **kw):
        '''
        Ignores filetype (since only one is supported, so we always know what it is), and kw (since there are no special behaviors).
        '''
        a = np.reshape(np.fromstring(stream.read(), np.int16), (-1, 1))
        fs = 400000.0
        a = a.astype(np.float64) / 32767
        d = gd.Doc({'fs': fs, 'data': a})
        return (d, None)


call1_p = CallParse()