#!/usr/bin/env python

import gdblocks.portfors.timingUI as ti
import matplotlib.pyplot as plt



def fig1(cells, **options):
	'''
	d: CellExp, fig: i, save: FileName? -> None (draws in MPL figure fig. Writes
		to file save, if save is specified)
	
	Make a raster plot of cell document d, using MPL figure fig. If save is a 
	string, save the figure to that file name. pass trng, or look at (0,200)
	
	'''
	x_range = options.get("x_range", (0, 200))
	condition = options.get("condition", "cond1")
	fig = plt.figure(options.get("fig", 1))
	spec_ar = options.get("spec_ar", .5)
	spec_npts = options.get("spec_npts", 1000)
	
	stimclasses = cells.values()[0]['stimclasses']
	stimfiles = dict([(k, [stimclasses[k]['file'], None]) for k in stimclasses if stimclasses[k]['type'] == 'stored file'])
	for k in stimfiles:
		stimfiles[k][1] = ti.make_thumbnail(stimfiles[k][0], ar=spec_ar, npts=spec_npts)
	plt.imshow(stimfiles.values()[0][1])
	
	if options.get("save"):
		plt.savefig(options["save"])

if __name__ == "__main__":
	fig1(ti.celldoc(ti.ALLEXP))