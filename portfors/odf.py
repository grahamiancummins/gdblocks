#!/usr/bin/env python
# encoding: utf-8
#Created by  on 2010-10-11.

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

from gicdat.base import Parser
import gicdat.doc
from gicdat.control import report
import zipfile, numpy
import gicdat.stdblocks.xmlparse as xp
import re

def tableCellContents(c):
	if c['_els']:
		v = c['_els'][0]['_cdata'].strip()
	elif 'office:value' in c:
		v= c['office:value']
	else:
		v= ''
	return v

def row2list(r):
	l = []
	mel = 0
	largest_real = 0
	for e in r['_els']:
		if not e['_tag'] == 'table:table-cell':
			report('warning: unexpected ods table element %s' % (str(e),))
			continue
		v = tableCellContents(e)
		nreps = int(e.get('table:number-columns-repeated', 1))
		for i in range(nreps):
			l.append(v)
		if v:
			mel = max(mel, len(v))
			largest_real = len(l)
	return (l[:largest_real], mel)

def table2array(d):
	rows = xp.quickSearch(d, 'table:table-row', 3, True)
	tab = []
	mel = 0
	mnc = 0
	mnr = 0
	ri = 0
	for r in rows:
		rl, melr = row2list(r)
		mel = max(mel, melr)
		mnc = max(mnc, len(rl))
		nreps = int(r.get('table:number-rows-repeated',1))
		for i in range(nreps):
			tab.append(rl)
		if rl:
			mnr = len(tab)
	if not all ([mel, mnc, mnr]):
		return numpy.zeros((0,0), str("|S1"))
	dts = "|S%i" % mel
	ta = numpy.zeros((mnr, mnc), str(dts))
	for i in range(mnr):
		for j, s in enumerate(tab[i]):
			ta[i,j] = s
	return ta

class ODSParse(Parser):
	'''	
	This parser handles opendocument spreadsheets (read only)

	'''
	canread = ('application/vnd.oasis.opendocument.spreadsheet',)
	extensions = {'ods': 'application/vnd.oasis.opendocument.spreadsheet'}

	def read(self, stream, filetype, **kw):
		f=zipfile.ZipFile(stream, 'r')
		ss = xp.quickRead(f.open('content.xml'))
		f.close()
		doc = gicdat.doc.Doc()
		for i, t in enumerate(xp.quickSearch(ss, 'table:table', 10, True)):
			dat = table2array(t)
			n = {}
			for key in t:
				if not key.startswith("_"):
					n[key] = t[key]
			n[''] = 'opendocument.spreadsheet.table'
			n['table'] = dat
			doc.set('sheet%i' % (i+1,),  n)
		return (gicdat.doc.Doc(doc), None)
	
def qtcast(v):
	try:
		v = int(v)
	except:
		try:
			v = float(v)
		except:
			v = str(v)
	return v	
	
class CSVParse(Parser):
	canread = ('application/csv',)
	canwrite = ('application/csv',)
	extensions = {'csv': 'application/csv'}
	
	def read(self, stream, ft, **kw):
		lines = stream.read().split('\n')
		lines = [map(qtcast, l.split(',')) for l in lines]
		if 'ignore' in kw:
			lines = lines[kw['ignore']:]
		if 'labels' in kw:
			labs = kw['labels']
		else:
			labs = lines[0]
			lines = lines[1:]
		if 'tag' in kw:
			tag = kw['tag']
			if tag.startswith('auto_'):
				tag = tag[5:]
				labs.append(tag)
				for i, l in enumerate(lines):
					l.append(i)
		else:
			tag = labs[0]
		g = self.fromtable(lines, labs, tag)
		return (g, None)
	
	def getpref(self, d):
		z = re.compile("(\D+)(\d+)$")
		ks ={}
		for k in d:
			m = z.match(k)
			if m:
				p = m.groups()[0]
				if not p in ks:
					ks[p] = 1
				else:
					ks[p] = ks[p]+1
		m= 0
		p = ''
		for k in ks:
			if ks[k]>m:
				p = k
				m = ks[k]
		return p
	
	def fromtable(self, t, labels=None, tag=None):
		if not labels:
			labels = list(t[0,:])
			t = t[1:,:]
		if not tag:
			tag = labels[0]
		inds = dict([(l, labels.index(l)) for l in labels])
		g = gicdat.doc.Doc()
		for l in t:
			n = "%s%s" % (tag, l[inds[tag]])
			if len(l)<len(labels):
				continue
			d = dict([(k, l[inds[k]]) for k in labels if not k == tag])
			g[n] = d
		return g
	
	def ss(self, s):
		s = str(s)
		if ',' in s:
			s = s.replace(',', ';')
		return s
	
	def write(self, d, stream, filetype, **kw):
		if 'prefix' in kw:
			p = kw['prefix']
		else:
			p = self.getpref(d)
		ks = [k for k in d.keys(sort=True) if k.startswith(p)]
		labels = set()
		for k in ks:
			labels.update(d[k].keys(-1, subdockeys=False))
		labels = sorted(labels)
		stream.write(','.join([p]+labels) + '\n')
		for k in ks:
			s = [k[len(p):]]+[self.ss(d[k][l]) for l in labels]
			stream.write(','.join(s)+'\n')
		
			
csv_p = CSVParse()
ods_p = ODSParse()
