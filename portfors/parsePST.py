#!/usr/bin/env python

#some information is copied from batgor-0.94 
#Copyright (c) 2005,2006 by Ed Groth,
#under the terms of the GNU General Public License
#but the majority of the parsing information is reimplemented based
# on the Portfors lab's Matlab parsers (authors unknown).

#New code copywrite Graham Cummins 2010

#This program is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 2 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program; if not, write to the Free Software
#Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

from gicdat.base import Parser
import gicdat.doc as gd

def asNumber(s):
	try:
		n = int(s)
	except ValueError:
		n = float(s)
	return n

stimtypes = {1:'tone',8:'stored file'}	

stimtypes = {1:'tone',
             2:'fmsweep',
             3:'synthesized_batsound',
             4:'amsound',
             5:'broad_band_noise',
             6:'narrow_band_noise',
             7:'click',
             8:'stored file',
             9:'high_pass_noise',
             10:'low_pass_noise',
             11:'sine_wave_modulation',
             12:'square_wave_modulation'};

def parseStim(l):
	stim = {}
	if l.startswith('0'):
		return stim
	vals = l.split()
	try:
		stim['type'] = stimtypes[int(vals[1])]
	except KeyError:
		print "this stimulus type isn't supported yet"
		raise
	stim['attenuation'] = float(vals[2])
	stim['duration'] = float(vals[3])
	stim['delay'] = float(vals[4])
	if stim['type'] == 'stored file':
		stim['reverse'] = int(vals[5])
		if not int(vals[6]):
			#using the "Old School" formatting?
			stim['file'] = vals[20]
		elif len(vals[24])>1 and vals[24]!='-1':
			stim['file'] = vals[24] + '.call1'
		else:
			stim['file'] = vals[33] + vals[35]
			
	elif stim['type']=='tone':
		stim['frequency'] = float(vals[5])
		stim['rise_fall'] = float(vals[6])
	else:
		print("This stimulus type (%s) isn't supported. Instead setting the 'value' attribute to the raw contents of the stimulus line" % stim['type'])
		stim['value'] = l
	return stim
	

def parseTrace(d):
	l = d['lines']
	tr = {}
	vals = map(asNumber, l[0].split())
	tr['nsweeps'] = vals[0]
	tr['samplerate_stim'] = vals[1]
	tr['samplerate_resp'] = vals[3]
	tr['duration'] = vals[4]
	tr['nsamples'] = int(round(vals[4]*vals[3]/1000))
	tr['nbytes'] = tr['nsamples']*tr['nsweeps']*2
	tr['stimulus']  = [parseStim(l[i]) for i in [4, 9, 14, 19]]
	tr['linenoinpst'] = d['lineno']
	return tr

def getTrace(sections):
	while sections[0]['type']!='trace parameters':
		print "Discarding section of type %s" % sections[0]['type']
		sections.pop(0)
		if not sections:
			return None
	tr = sections.pop(0)
	trace = parseTrace(tr)
	if sections[0]['type'] == 'spike data':
		trace['spikes'] = [map(int, l.split()[1:]) for l in sections[0]['lines']]
		sections.pop(0)
	return trace

def getTest(sections):
	while sections[0]['type']!='test parameters':
		print "Discarding section of type %s on line %i" % (sections[0]['type'], sections[0]['lineno'])
		sections.pop(0)
		if not sections:
			return None
	te = sections.pop(0)
	t = te['lines']
	l3 = t[2].split()
	test = {'testtype':t[0], 'time':t[1], 'size':int(l3[0]), 'scan':' '.join(l3[1:]), 'traces':[], 'comment':te['comment']}
	while sections and sections[0]['type']!='test parameters':
		test['traces'].append(getTrace(sections))
	if len(test['traces'])<test['size']:
		test['complete'] = False
	elif len(test['traces'])>test['size']:
		print "Surprise: test seems to have too many traces"
		print te['lineno'], test['size'], len(test['traces'])
	else:
		test['complete'] = True
	return test

def addOffsets(tests):
	currentOffset = 0
	for t in tests:
		for tr in t['traces']:
			tr['raw_data_offset'] = currentOffset
			currentOffset += tr['nbytes']

def parse(stream):
	lines = stream.readlines()
	sections =[{'lines':[], 'type':None}]
	commenting = [False, 0]
	for i, l in enumerate(lines):
		if commenting[0]:
			sections[commenting[1]]['comment'] = l
			#print('commenting %i with %s' % (commenting[1], l))
			commenting[0] = False
			
		elif l.startswith('End of '):
			#print(l, i+2, commenting)
			ty =  l[7:].strip()
			if ty == 'auto test':
				commenting[0] = True
				sections[-1]['lineno']+=2
				continue
			if ty == 'test parameters':
				commenting[1] = len(sections)-1
				#print "test starts on line %i" % (sections[-1]['lineno'])
			sections[-1]['type'] = ty
			sections.append({'lines':[], 'lineno':i+2,'type':None})
		else:
			sections[-1]['lines'].append(l.strip())
	if sections[-1]['type'] == None:
		sections.pop(-1)
	if sections[0]['type'] == 'ID information':
		meta = sections[0]['lines']
		sections.pop(0)
	else:
		print "Surprise, first section isn't ID info!"
		meta = ''
	tests = []
	ind = 1
	while sections:
		tests.append( getTest(sections))
		
	if tests[-1] == None:
		tests.pop(-1)
	addOffsets(tests)	
	return (meta, tests)

def tupulate(lol):
	if type(lol) != list:
		return lol
	return tuple([tupulate(l) for l in lol])

class PSTParse(Parser):
	'''	
	Batlab PST (text-based experimental results) files

	'''
	canread = ('application/batlab.pst',)
	extensions = {'pst': 'application/batlab.pst'}

	def read(self, stream, filetype, **kw):
		comments, l = parse(stream)
		doc = gd.Doc()
		metas = ['filename', 'date', 'experiment_type', 'investigator', 'machine']
		for i in range(len(comments)):
			if i < len(metas):
				doc['metadata.' + metas[i]] = comments[i]
			else:
				doc['metadata.line%i' % i] = comments[i]
		for tei in range(1, len(l)+1):
			tn = 'test%i' % tei
			d = l[tei-1]
			for k in d:
				if k!='traces':
					doc[tn+'.'+k] = d[k]
			doc[tn+'.tag']='test'
			doc[tn+'.index'] = tei
			for tri in range(1, len(d['traces'])+1):
				trn = 'test%i.trace%i' % (tei, tri)
				trd = d['traces'][tri-1]
				for k in trd:
					if not k in ['spikes', 'stimulus']:
						doc[trn+'.'+k] = trd[k]
				doc[trn+'.index'] = (tei, tri)
				doc[trn+'.tag']='trace'
				doc[trn+'.samplerate']=1e6   # batlab always reports in microseconds
				doc[trn+'.start']=0.0
				doc[trn+'.events'] = trd['spikes']  
				for si, sd in enumerate(trd['stimulus']):
					if sd:
						sn = trn+'.stim.ch%i' % si
						for k in sd:
							doc[sn+'.'+k] = sd[k]
		return (doc, None)

pst_p =  PSTParse()

if __name__=='__main__':
	import sys
	g = parse(sys.argv[1])
	for s in g:
		print s['type']

