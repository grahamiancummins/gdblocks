#!/usr/bin/env python -3
# encoding: utf-8

#Created by gic on  Tue Dec  7 13:18:39 CST 2010

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

import matplotlib.pyplot as plt
import numpy as np
from gicdat.gui.mpl import COLORS, LSTYLES
import StringIO, Image, ImageOps


def _gplot(d, groups, atrs, gids, glen):
	gmemb =dict([(i, []) for i in gids])
	for ri, gid in enumerate(groups):
		gmemb[gid].append(ri)
	if atrs:
		bstart = 0
		dblock = False
		lattr = atrs[0]
	for i, gid in enumerate(gids):
		col = COLORS[i]
		if i%2:
			plt.axhspan(glen*i, glen*(i+1)-1, facecolor=(.8,.8,1), alpha=0.3, edgecolor='none')
		for pi, ri in enumerate(gmemb[gid]):
			ylev =  i*glen + pi
			if atrs and lattr!=atrs[ri]:
				if dblock:
					plt.axhspan(bstart, ylev, facecolor='0.1', alpha=0.1, edgecolor='none')
				dblock = not dblock
				bstart = ylev
				lattr = atrs[ri]
			x = np.array(d[ri]) 
			y = np.ones(x.shape[0]) + ylev		
			plt.plot(x, y, marker='.',color=col, linestyle='None')
	return len(gids)*glen


def make_silloette(data, npts, col=(.8, .8, 1), **kwargs):
	ppp = np.floor(data.shape[0]/npts)
	if ppp<2:
		plt.plot(data)
	else:
		clip=data.shape[0]%ppp
		if clip:
			data = np.concatenate([data, np.ones(ppp-clip)*data[-1]])
		data=np.reshape(data, (-1, ppp))
		ymax = np.maximum.reduce(data, 1)
		ymin = np.minimum.reduce(data, 1)
		x= np.arange(len(ymax))
		plt.fill_between(x, ymax, ymin, color=col)

def make_spectrogram(dat, npts, fs=333333.3, winwidth=None, 
                     window=None, fstart=0, fstop=120000, fstep=5000,
                     cmap = 'gist_heat_r', **kwargs):
	tstep =int(round(dat.shape[0]/float(npts)))
	winwidth = winwidth or tstep
	if window:
		windowa = np.linspace(0, 1, winwidth)
		windowa = np.concatenate([window, window[::-1]])
	nft = winwidth + 1 
	ind = np.arange(fstart, fstop, fstep)
	ind = nft * ind / (fs/2.0)
	ind = ind.astype(np.int32)[::-1]
	xind = np.arange(winwidth, dat.shape[0]-winwidth, tstep).astype(np.int32)
	ft = np.zeros((xind.shape[0], ind.shape[0]))
	for i in range(xind.shape[0]):
		d = dat[xind[i]-winwidth:xind[i]+winwidth]
		#warning, very strange behavior of rfft for 2d arrays,
		#even if shape is Nx1 or 1xN.
		if window:
			d = d * windowa
		d = np.fft.rfft(d)
		ft[i,:] = abs(d[ind][range(d[ind].shape[0]-1,-1,-1)])	
	plt.imshow(ft.transpose(), aspect='auto')
	if cmap:
		plt.set_cmap(cmap)
	return ft

THUMBS = {'silloette':make_silloette,
          'spectrogram':make_spectrogram}

def make_thumbnail(data, mode='silloette', npts=500,
                   ar=.15,show=False, **kwargs):
	y = int(round(npts*ar))
	w = npts/100
	#plt.rc('axes', linewidth=0)
	f = plt.figure(11, figsize=(w, w*ar), dpi = 100)
	plt.subplot(111, frameon=False)
	THUMBS[mode](data, npts, **kwargs)
	a = f.get_axes()[0]
	a.xaxis.set_visible(False)
	a.yaxis.set_visible(False)
	f.subplots_adjust(left=0, right=1, bottom=0, top = 1)
	f.canvas.draw()
	s = StringIO.StringIO()
	f.savefig(s, format=kwargs.get('format', 'png'))
	s.seek(0)
	open('test.pdf', 'wb').write(s.read())
	s.seek(0)
	im = ImageOps.flip(Image.open(s))
	if not show:
		plt.close()
	return im 


def _ixfill(a, n, v=255):
	return np.zeros( (a.shape[0], n, a.shape[2]), a.dtype)+v

def _xwinimg(a, o, w):
	ipu = a.shape[1]/float(o[1])
	ws = int(round( ipu*(w[0]-o[0])))
	npix = int(round( ipu*(w[1]-w[0])))
	if ws < 0:
		a=np.column_stack([_ixfill(a,-ws), a ])
	elif ws > 0:
		a = a[:,ws:,:]
	if npix > a.shape[1]:
		a=np.column_stack([a, _ixfill(a,npix-a.shape[1])])
	else:
		a = a[:,:npix,:]
	return a

def xwin_img(i, o, w):
	"""
	i:PIL Image, o: (start:x, dur:x), w: WinSpec -> PIL Image
	"""
	a = np.array(i)
	ww = [_xwinimg(a, o, win) for win in w]
	a = np.column_stack(ww)
	return Image.fromarray(np.column_stack(ww))

class RasterGrid(object):
	def __init__(self, h, w, fig=1, tscl=1000.0):
		self.w = w 
		self.h = h
		self.fig = fig
		self.f = plt.figure(fig)
		self.n =0
		self.tmin = np.inf
		self.tmax = 0
		self.vpad = .3
		self.tscl = tscl
		self.hltcol = (.8,.8,1)
		self.misscol = (1,.8,.8)
		self.yticks = [[], []]
		self.titles = [['']*self.w for _ in range(self.h)]
		
		
	def clear(self):
		f=plt.figure(self.fig)
		plt.clf()
		self.yticks = [[], []]
		f.canvas.draw()
		
	def add(self, grid, col=0, hlt =False, name=''):
		plt.figure(self.fig)
		mnew = max( [max(map(len, l)) for l in grid]   )
		for h in range(self.h):
			for w in range(self.w):
				self.sp(h,w)
				if hlt:
					bot = self.n+self.vpad
					top = self.n+mnew - self.vpad
					if 'thumbnail' in hlt:
						if 'tnbounds' in hlt:
							tmi, tma = hlt['tnbounds']
						else:
							tmi, tma = self.tmin, self.tmax
						plt.imshow(hlt['thumbnail'], 
						           extent = (tmi, tma, bot, top),
						           aspect='auto')
					if 'roi' in hlt:
						alph = hlt.get('alpha', .5)
						hcol = hlt.get('color', self.hltcol)
						for rn in hlt['roi']:
							plt.fill((rn[0], rn[0], rn[1], rn[1]), 
								(bot, top, top, bot),
							    facecolor=hcol, alpha=alph, linewidth=0)
				for i, elist in enumerate(grid[h][w]):
					if len(elist)==0:
						continue
					x = np.array(elist)/self.tscl
					self.tmin = min(self.tmin, x.min())
					self.tmax = max(self.tmax, x.max())
					y = np.ones_like(x)*(self.n+i+1)
					plt.plot(x, y, marker='.',color=COLORS[col], linestyle='None')
				mthis = len(grid[h][w])
				if mthis+1<mnew:
					plt.axhspan(self.n+mthis+.5, self.n+mnew-.5, facecolor=self.misscol, alpha=.8, 
							linewidth=0, edgecolor='none')
		self.yticks[0].append(self.n)
		if name:
			self.yticks[1].append(name)
		else:
			self.yticks[1].append(str(self.n))
		self.n += mnew
				
	def sp(self, h, w):
		i = self.w*h+w +1
		return plt.subplot(self.h, self.w, i)
				
	def draw(self):
		f=plt.figure(self.fig)
		f.subplots_adjust(left=.04, right=.99, bottom=.03, top = .94, wspace=.05)
		for h in range(self.h):
			for w in range(self.w):
				sp = self.sp(h,w)
				tpad = .01*(self.tmax-self.tmin)
				plt.xlim([self.tmin-tpad, self.tmax+tpad])
				plt.ylim([-1, self.n+1])
				if sp.is_first_col():
					sp.set_yticks(self.yticks[0])
					sp.set_yticklabels(self.yticks[1])
				else:
					sp.yaxis.set_visible(False)
				if not sp.is_last_row():
					sp.xaxis.set_visible(False)
					#plt.gca().set_xticklabels([])
				if self.titles[h][w]:
					plt.title(self.titles[h][w])
		f.canvas.draw()
		
			
def plotScanAll(d, r = (1000, 50000, 2000), cnames = ("Control", "Bic&Str")):
	x = apply(range, r)
	f = plt.figure(1)
	plt.clf()
	lhr = {}
	names = []
	for i in range(len(d)):
		for j, k in enumerate(sorted(d[i])):
			dat = np.array(d[i][k])
			pname = "%s - %s" % (str(k), cnames[i])
			ph = plt.errorbar(x, dat[:,0], dat[:,1], color=COLORS[i], fmt=LSTYLES[j])
			names.append(pname)
			lhr[pname] = ph[0]
	plt.figlegend([lhr[n] for n in names], names, 'upper right')
	f.canvas.draw()

def dot2img(s):
	'''
	takes a string containing graphviz dot instructions, and returns a PIL 
	image object representing the graph (provided PIL is available). The image
	is flipped vertically, so it can be plotted with imshow
	
	'''
	import PIL.Image
	import PIL.ImageOps
	import os
	import tempfile
	fd, dotname = tempfile.mkstemp(".dot")
	os.close(fd)
	fd, pngname = tempfile.mkstemp(".png")
	os.close(fd)
	open(dotname, 'w').write(s)
	os.system("dot -Tpng -o %s %s" % (pngname, dotname))
	im = PIL.Image.open(pngname)
	imflip = PIL.ImageOps.flip(im)
	os.unlink(dotname)
	os.unlink(pngname)
	return imflip
	

