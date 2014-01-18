#!/usr/bin/env python -3
# encoding: utf-8

#Created by Graham Cummins on Apr 20, 2011

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
from gicdat.base import Parser
import gicdat.enc
import zipfile, StringIO
import Crypto.Cipher.AES as AES
import Crypto.Hash.SHA as SHA
import gicdat.control


def _key(pw):
    h = SHA.new()
    h.update(pw)
    return h.digest()[:16]


def _pad(s, l=16, c=' '):
    np = l - divmod(len(s), l)[1]
    return s + c * np


def enc(s, pw):
    e = AES.new(_key(pw))
    return e.encrypt(_pad(s))


def dec(s, pw):
    e = AES.new(_key(pw))
    return e.decrypt(s).rstrip(' ')


class GicParse(Parser):
    '''
    This parser handles gicdat's internal file format, which is a form of zip
    file

    '''

    canread = ('application/com.symbolscope.gicdat.enc',)
    canwrite = ('application/com.symbolscope.gicdat.enc',)
    extensions = {'gdenc': 'application/com.symbolscope.gicdat.enc'}

    def read(self, stream, filetype, **kw):
        if 'password' in kw:
            pw = kw['password']
        else:
            pw = gicdat.control.asksecret('password?')
        z = zipfile.ZipFile(stream, 'r')
        rdat = z.read('data.raw')
        rdat = dec(rdat, pw)
        dd = eval(z.read('doc.py'))
        dd = dec(dd, pw)
        doc = gicdat.enc.dict2doc(dd, rdat, False)
        return (doc, None)

    def write(self, d, stream, filetype, **kw):
        if 'password' in kw:
            pw = kw['password']
        else:
            pw = gicdat.control.asksecret('password?')
        z = zipfile.ZipFile(stream, 'w')
        dstr = StringIO.StringIO()
        d = gicdat.enc.doc2dict(d, dstr)
        z.writestr('doc.py', enc(repr(d), pw))
        z.writestr('data.raw', enc(dstr.getvalue(), pw))
        dstr.close()


gic_p = GicParse()
