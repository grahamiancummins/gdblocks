#!/usr/bin/env python

import os, re
from gicdat.control import report
import gicdat.io as gio
import gicdat.doc as gd
import numpy as np
import gicdat.search as gds
from gicdat.control import report
from gdblocks.istac.acell import hist
from gicdat.tag import Tag

basedir =  'project/christine'
dsheet_path = 'DataSheet.ods'
basedir = os.path.join(os.environ['HOME'],basedir)
dsheet_path = os.path.join(basedir, dsheet_path)
callsdir = os.path.join(basedir, 'CBA calls Adults')
pathprefix = "Mouse "

cellIdLab = 'Cell ID'
mouseNoLab = 'Mouse #'
testLab = 'Test #'
conditionLab = 'Bic + Strych'

#KEEPMETA = [
#'test.' + mouseNoLab,
#'test.' + conditionLab,
#'test.' + cellIdLab,
#'stimulus.file',
#'stimulus.frequency',
#'test.Time from drug app',
#'stimulus.attenuation',
#'test.dB change',
#'test.BF',
#'stimulus.reverse',
#'stimulus.delay',
#'test.Eject current',
#'stimulus.rise_fall'
#]


def getDsheet(dsheet=None):
	if not dsheet:
		dsheet = dsheet_path
	tab = gio.read(dsheet)['sheet1.table']
	lab = dict([(str(tab[0,i]), i) for i in range(tab.shape[1]) if tab[0,i]])
	files = {}
	for row in range(1, tab.shape[0]):
		cid = tab[row, lab[	cellIdLab]].strip()
		if cid in ['', 'Not complete']:
			continue
		mouse = tab[row, lab[mouseNoLab]]
		if not mouse in files:
			files[mouse] = []
		files[mouse].append( dict([(l, tab[row, lab[l]]) for l in lab]))
	return files

def numOrRan(s):
	m = re.search("(\d+)-?(\d+)?", s)
	start, stop = m.groups()
	if not stop:
		return [int(start)]
	else:
		return range(int(start), int(stop)+1)

def getTests(dsheet=None, bdir = basedir):
	files = getDsheet(dsheet)
	doc = gd.Doc()
	testk = re.compile('test(\d+)')
	for f in files:
		ns = "track%s" % f	
		mouse = numOrRan(f)[0]
		dfpath = os.path.join(bdir, pathprefix+str(mouse), pathprefix+f)
		pstpath = os.path.join(dfpath, pathprefix+f+'.pst')
		if not os.path.isfile(pstpath):
			report("No data for %s" % f)
			continue
		else:
			report ("found data for %s" % f)
		ld = gio.read(pstpath)
		tids = {}
		for tests in files[f]:
			ids = numOrRan(tests[testLab])
			for i in ids:
				tids[i] = tests
		found = set()
		for k in ld:
			m = testk.match(k)
			if m:
				i = int(m.groups()[0])
				if i in tids:
					doc[ns+'.'+k] = ld[k]
					found.add(i)
		for i in found:
			n = ns + '.test%i.' % i
			for k in tids[i]:
				doc[n + k] = tids[i][k]	
	return doc

def sweepArray(d, ns, maxns):
	sa = np.zeros((ns, maxns)).astype(np.int32) -1
	for i in range(ns):
		spikes = d[d[:,1]==i,0]
		sa[i,:spikes.shape[0]] = spikes
	return sa

def mdatk(doc):
	vals = {}
	for k in doc:
		b = k.split('.')[-1]
		v = doc[k]
		if not b in vals:
			vals[b] = set()
		vals[b].add(v)
	for k in vals.keys():
		if len(vals[k]) <2:
			del(vals[k])
		elif len(vals[k])>50:
			del(vals[k])
	return vals
	
def up(k, n):
	l =k.split('.')
	return '.'.join(l[:-n])

def node(doc, k):
	d = {}
	bk = up(k, 1)
	tk =up(bk, 1)
	d['drug'] = doc[tk+'.'+conditionLab].lower().startswith('y')
	d['cell'] = int(doc[tk+'.'+cellIdLab])
	d['attn'] = int(doc[bk+'.stim.ch0.attenuation'] or 0.0)
	if doc[bk+'.stim.ch0.file']:
		d['tone'] = False
		d['stim'] = doc[bk+'.stim.ch0.file']
	else:
		d['tone'] = True
		d['stim'] = int(doc[bk+'.stim.ch0.frequency'] or 0)
	d['evts'] =doc[k]
	return d

def reducedDoc(doc):
	dats = [k for k in doc if k.endswith('events')]
	nodes = [node(doc, k) for k in dats]
	vals =  ['drug', 'cell', 'tone', 'stim', 'attn']
	#cols = gds.pat(dict([(k, gds.Collector()) for k in vals]))
	#__ = [cols == n for n in nodes]
	d = gd.Doc()
	for i, n in enumerate(nodes):
		ns = 'n%i' % i
		for k in n:
			d[ns+'.'+k] = n[k]
	return d


def sgroups(doc, cell=1, tone=False):
	tk = doc.topkeys('')
	t = Tag({'cell':cell, 'tone':tone})
	keys = doc.tags(t, tk)
	evts = [(doc[k+'.drug'], doc[k+'.stim'], doc[k+'.evts']) for k in keys]
	stims = sorted(set([e[1] for e in evts]))
	di = []
	do = []
	ni = []
	no = []
	for s in evts:
		sid = stims.index(s[1])
		for r in s[2]:
			if s[0]:
				di.append(sid)
				do.append(r)
			else:
				ni.append(sid)
				no.append(r)
	d = gd.Doc()
	d['n.i'] = np.array(ni) 	
	d['n.o'] = tuple(no) 	
	d['d.i'] = np.array(di) 	
	d['d.o'] = tuple(do) 	
	return d 

def _wrtcsv(isp, osp, fn):
	f = open(fn+"_trials.csv", 'w')
	f2 = open(fn+"_spikes.csv", 'w')
	f.write('trial, stim\n')
	f2.write('trial, spiketime\n')
	for i in range(isp.shape[0]):
		f.write('%i, %i\n' % (i, isp[i]))
		for j in osp[i]:
			f2.write("%i, %i\n" % (i, j))
	f.close()
	f2.close()

def sgroups2csv(sgs, fn):
	_wrtcsv(sgs['n.i'], sgs['n.o'], fn+'nodrug')
	_wrtcsv(sgs['d.i'], sgs['d.o'], fn+'withdrug')

def get(opts, args):
	'''
	get loads all data specified in a data sheet from a set of PST files, and
	saves the resulting document to an output file. Usage is:
	
	get [-d path] [-s path] outfilename

	outfilename specifies where to save the output. The type of this file is
	inferred from its extension (if there is no "." in outfilename, ".gic" is
	used by default). 

	-d path specifies to use a data directory of "path". The various "Mouse..."
	directory trees containing PST files should be children of path

	-s path specifies to use a datasheet (ods spreadsheet) that is stored at
	path. 

	by default, -d and -s paths use the module level parameters basedir and
	dsheet_path (which currently assume that you have system with a HOME, the
	project directory is ~/project/christine, and the datasheet is in this
	directory, and is named DataSheet.ods) .

	'''
	bdir = opts.get('d') or basedir
	dsheet = opts.get('s') or dsheet_path
	fname = args[0]
	if not '.' in fname:
		fname = fname + ".gic"
	doc = getTests(dsheet, bdir)
	gio.write(doc, fname)

def reduce(opts, args):
	'''
	Generates a document with a reduced representation of cell I/O properties,
	and save this to disk. Usage is:

	reduce [-d path] outfilename

	outfilename specifies where to save the output. The type of this file is
	inferred from its extension (if there is no "." in outfilename, ".gic" is
	used by default). 

	If -d is specified, 'path' should be a file path to a document of the 
	sort written by the "get" subcommand. Specifying this saves the time 
	required to do the raw data extraction, and is also required if you 
	wanted to pass non-default arguments to get. If -d is ommitted, the
	data document is constructed as though you called "get" with default
	options (but this document is not saved to a file)

	'''
	fname = args[0]
	if 'd' in opts:
		doc = gio.read(opts['d'])
	else:
		doc = getTests()
	doc = reducedDoc(doc)
	if not '.' in fname:
		fname = fname + ".gic"
	gio.write(doc, fname)

def csv(opts, args):
	'''
	Write processed cell data to csv files. 

	usage is:
	csv [-d path] [-t] cid [cid ...]

	If -d is specified, 'path' should be a file path to a document of the 
	sort written by the "get" or "reduce" subcommand. Specifying this saves the
	time required to do the raw data extraction, and is also required if you
	wanted to pass non-default arguments to get. If -d is ommitted, the data
	document is constructed as though you called "get" and then "reduce" with
	default options (but this document is not saved to a file). If -d is
	specified, this function will check for an element "params" in the
	document. If it is present, the document is assumed to be reduced.
	Otherwise, it is assumed to be raw. This works if the document was
	generated by this module, but may not always be safe. If you have problems,
	call "reduce" explicitly to produce the target document.

	If -t is specified, the data will be for "tone" stimuli (tuning 
	curve type tests), and if not, the data will include controls
	and tests using file type stimuli (calls).

	The "cid"s should be numbers, such as 1, 2 , 3. You may specify any 
	list of them. The result will be 4 csv files written to disk in the 
	current directory for each number. These will have names:
	cellNnodrug_spikes.csv, cellNnodrug_trials.csv, cellNwithdrug_spikes.cvs, 
	and cellNwithdrug_trials.csv, where "N" is the cid.

	The "withdrug" and "nodrug" files specify different experimental
	conditions ("with" contains data from tests where conditionLab is "yes"
	and "no" contains test where it was "no". conditionLab is currently
	'Bic + Strych'). The "trials" files have exactly one line for each trial
	recorded. This contains, in the first column, the trial ID (a number that
	should increase by exactly 1 on each successive line), and in column 2 the
	stimulus ID used in that trial (a number that should be between 0 and 16
	inclusive, is not monatonically increasing, and will probably remain the
	same for 20 or 30 consecutive trials).

	The "spikes" files contain one line for each recorded spike, and
	contain, in the first column, the trial ID (which corresponds to the
	same number in the trials file, so allows you to look up the stimulus
	used), and in the second column the spike time (in microseconds after
	recording onset).	

	'''
	tones = 't' in opts
	if 'd' in opts:
		doc = gio.read(opts['d'])
	else:
		doc = getTests()
	if not 'n0.evts' in doc:
		doc = reducedDoc(doc)
	for cid in args:
		cid = int(cid)
		sgs = sgroups(doc, cid, tones)
		sgroups2csv(sgs, "cell" + str(cid)) 
		
def confusionDS(c, n=10000):
	'''
	draw samples from a confusion matrix
	'''
	rm = c.sum(1)
	isp = []
	osp = []
	for i in range(n):
		ine = drawpd(rm)
		oute = drawpd(c[ine,:])
		isp.append(ine)
		osp.append(oute)
	return (np.array(isp), np.array(osp))


if __name__=='__main__':
	from gicdat.cli.switches import cldispatch
	cldispatch([get, reduce, csv], 'd:s:t')
