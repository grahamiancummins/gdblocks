#!/usr/bin/env python

PDF = False

import matplotlib, os

if PDF:
    matplotlib.use('pdf')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import portfors.vis as vis
import portfors.timingUI as ti
import gicdat.io as gio
import numpy as np

OUTDIR = os.path.expanduser("~/Desktop/portfors")
DEFAULTS = {
    "x_range": (0, 200),
    "condition": "cond1",
    "fig": 1,
    "spec_ar": .5,
    "spec_npts": 2048,
    "time_window": (0, 200),
    "imformat": 'png',
    "rast_fs": 1000.0,
    "rast_color": "k",
    "rast_msize": 16.0,
    "png_dpi": 300,
    "fig_width": 20,
}
if PDF:
    DEFAULTS["imformat"] = "pdf"

SPEC = {
    "fs": 333333.3,
    "fstart": 0,
    "fstop": 120000,
    "fstep": 5000,
    "cmap": 'gist_heat_r',
    "winwidth": None,
    "window": None,
}


def get_fig():
    w = DEFAULTS['fig_width']
    dpi = DEFAULTS['png_dpi']
    ar = DEFAULTS['spec_ar']
    f = plt.figure(1, figsize=(w, w * ar), dpi=dpi)
    plt.clf()
    f.canvas.draw()
    return f


def save_spectrogram(stim, stim_info, npts=500, ar=.15, time_window=(0, 200), save="pdf"):
    infilename = os.path.join(ti.STIMDIR, stim_info['file'])
    stim_name = os.path.splitext(stim_info['file'])[0]
    data = gio.read(infilename)['data'][:, 0]
    fs = SPEC['fs'] / 1000.0
    lead_pad = np.zeros((0 - time_window[0]) * fs)
    tail_pad = np.zeros((time_window[1] - stim_info['duration']) * fs)
    data = np.concatenate([lead_pad, data, tail_pad])
    y = int(round(npts * ar))
    f = get_fig()
    plt.subplot(111, frameon=False)
    vis.make_spectrogram(data, npts, **SPEC)
    a = f.get_axes()[0]
    a.xaxis.set_visible(False)
    a.yaxis.set_visible(False)
    f.subplots_adjust(left=0, right=1, bottom=0, top=1)
    f.canvas.draw()
    outfile = os.path.join(OUTDIR, "{}_{}_spectrogram.{}".format(stim_name, stim, save))
    f.savefig(open(outfile, 'wb'), format=save)
    return outfile


def save_raster(file_name, evts, **options):
    f = get_fig()
    for i in range(len(evts)):
        x = np.array(evts[i]) / options["rast_fs"]
        y = np.ones_like(x) * (i + 1)
        plt.plot(x, y, marker='.', color=options['rast_color'], linestyle='None', markersize=options["rast_msize"])
    plt.xlim(options["time_window"])
    plt.ylim([0, len(evts) + 1])
    f.canvas.draw()
    f.savefig(open(file_name, 'wb'), format=options['imformat'])


def save_rasters(cell_name, evts, stims, **options):
    for stim in sorted(set(stims)):
        fn = os.path.join(OUTDIR, "{}_stim{}.{}".format(cell_name, stim, options['imformat']))
        s_evts = [evts[i] for i in range(len(evts)) if stims[i] == stim]
        save_raster(fn, s_evts, **options)


def save_all_frames(cells, **options):
    o = {}
    o.update(DEFAULTS)
    o.update(options)
    wrote_stims = []
    for cell_name in cells:
        stimclasses = cells[cell_name]['stimclasses']
        for sn in stimclasses:
            if stimclasses[sn]['type'] == 'stored file' and not sn in wrote_stims:
                save_spectrogram(sn, stimclasses[sn], o['spec_npts'], o['spec_ar'], o['time_window'], o['imformat'])
                wrote_stims.append(sn)
        save_rasters(cell_name, cells[cell_name]['cond1']['evts'], cells[cell_name]['cond1']['stims'], **o)


def fig1(cells, **options):
    '''
    d: CellExp, fig: i, save: FileName? -> None (draws in MPL figure fig. Writes
        to file save, if save is specified)

    Make a raster plot of cell document d, using MPL figure fig. If save is a
    string, save the figure to that file name. pass trng, or look at (0,200)

    '''

    stimclasses = cells.values()[0]['stimclasses']
    for k in stimfiles:
        stimfiles[k][1] = ti.make_thumbnail(stimfiles[k][0], ar=spec_ar, npts=spec_npts, format='pdf')
    plt.imshow(stimfiles.values()[0][1])

    if options.get("save"):
        plt.savefig(options["save"], format='pdf')


if __name__ == "__main__":
    cells = ti.celldoc(ti.ALLEXP)
    #fig1(cells, save="test_portfors_fig.pdf")
    save_all_frames(cells)