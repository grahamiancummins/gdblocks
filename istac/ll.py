#!/usr/bin/env python -3
# encoding: utf-8

#Created by gic on DATE.

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

#!/usr/bin/env python
import numpy as np
import gwn, istac
import numpy.linalg as la
import rand

LENGTH = 60
LEAD = 40
TESTPROP = .2
BOOTSTRAP = 10
UCSESIZE = 15000


def logLike(mu, sig, ens):
    '''
    Evaluate the log likelyhood that the data in ens (column-format
    <array[M,N]>) were drawn from the Gaussian model specified by mu (mean,
    <array[M]>, and sig (covariance, array[M,M] )

    Return is a float, the log likelyhood.

    '''
    nd = sig.shape[0] / 2.0
    norm = 1.0 / (np.sqrt(np.linalg.det(sig)) * (2 * np.pi) ** nd)
    lnorm = np.log(norm)
    icov = np.linalg.inv(sig)
    dist = 0
    for i in range(ens.shape[1]):
        x = ens[:, i] - mu
        out = -.5 * np.dot(x, np.dot(icov, x))
        dist += out
    dist = lnorm + dist / ens.shape[1]
    return dist


def splitEns(ens, prop, rs):
    '''
    Randomly split the ensemble ens <array[M,N]> into two parts. The return is
    < (array[M, N-L, array[M, L]) >. The parameter "prop" <float on (0,1)>
    determines the proportion of columns in the second part. L =
    int(round(M*prop))

    rs <int> determines the random seed state
    '''
    inds = rand.testset(ens.shape[1], rs)
    prop = int(round(ens.shape[1] * prop))
    return (ens[:, inds[:-prop]], ens[:, inds[-prop:]])


def sigLength(ens, ec):
    #FIXME: implement this
    pass


def ece(stim, evts, length=LENGTH, lead=LEAD):
    '''
    return an event conditioned ensemble containing windows drawn from the
    stim, conditioned on events in the sequence evts.

    length and lead determine the size of the window	(each
    window is length samples long, and starts lead samples before the
    event).

    The returned ensemble uses columns as elements, and rows as samples, so
    it is < array[length,len(evts)] >.

    '''
    ens = []
    for e in evts:
        if e - lead < 0 or e - lead + length >= stim.size:
            continue
        ens.append(stim[e - lead:e - lead + length])
    return np.column_stack(ens)


def ucse(stim, size=UCSESIZE, length=LENGTH):
    '''
    return an ensemble <array[length, size]> of "size" length "length"
    windows chosen randomly from stim. Alternately, if size is a False
    value, return <array[length, stim.size -length]> containing
    all possible windows.

    '''
    if size:
        revts = rand.ucevts(stim.size - length, size, 0)
        revts += length
    else:
        revts = np.arange(length, stim.size)
    ens = [stim[i - length:i] for i in revts]
    return np.column_stack(ens)


def eqrc(rcses):
    eqrcs = []
    nsamp = min([r.shape[1] for r in rcses])
    for i, r in enumerate(rcses):
        ind = rand.ucevts(r.shape[1], nsamp, 5 + i)
        eqrcs.append(r[:, ind])
    rc = np.column_stack(eqrcs)
    return rc


def lltest(rc1, rc2, testprop=TESTPROP, bootstrap=BOOTSTRAP):
    llv = np.zeros((2, 2, bootstrap))
    for bs in range(bootstrap):
        rc1m, rc1t = splitEns(rc1, testprop, 2 * bs)
        rc2m, rc2t = splitEns(rc2, testprop, 2 * bs + 1)
        m1mu = rc1m.mean(1)
        m2mu = rc2m.mean(1)
        m1sig = np.cov(rc1m)
        m2sig = np.cov(rc2m)
        if rc1.shape[0] == 1:
            m1sig = np.array([[m1sig.flat[0]]])
            m2sig = np.array([[m2sig.flat[0]]])
        #print m1mu.shape, m1sig.shape
        llv[0, 0, bs] = logLike(m1mu, m1sig, rc1t)
        llv[0, 1, bs] = logLike(m1mu, m1sig, rc2t)
        llv[1, 0, bs] = logLike(m2mu, m2sig, rc1t)
        llv[1, 1, bs] = logLike(m2mu, m2sig, rc2t)
    ll = np.concatenate([llv.mean(2)[..., np.newaxis], np.std(llv, 2)[..., np.newaxis]], 2)
    return ll


def is_comp(ce, uc, clevel):
    iss = istac.istacspace(ce, uc, clevel)['vecs']
    return iss.transpose()


def is_comp_r(ce, uc, clevel):
    iss = istac.istacspace(ce, uc, clevel, True)['vecs']
    return iss.transpose()


def is_comp_old(ce, uc, clevel):
    iss = istac.istacspaceOld(ce, uc, clevel)['vecs']
    return iss.transpose()


def rCompress(comp):
    def cf(rc, uc, cl):
        fd = rc.shape[1]
        vecs = comp(rc, uc, fd)
        return vecs[-cl:, :]

    return cf


def siglength_comp(rc1, rc2, uc, clevel):
    xmin, xmax = sigLength(uc, ce)
    space = np.eye(rc1.shape[0])[xmin:xmax, :]
    return space


def pca_space(ce, uc, clevel):
    '''
    PCA eigenvectors plus mean
    '''
    m = ce.mean(1)
    cov = np.cov(ce)
    val, vec = la.eig(cov)
    if clevel >= 2:
        space = np.column_stack([m, vec[:, :int(clevel) - 1]])
    elif clevel < 1:
        val = val / val.sum()
        howmany = np.nonzero(np.cumsum(val) >= clevel)[0][0]
        space = np.column_stack([m, vec[:, :howmany]])
    else:
        space = np.reshape(m, (-1, 1))
    return space.transpose()


def ppca_space(ce, uc, clevel):
    '''
    Pure PCA (eigenvectors only, no mean
    '''
    cov = np.cov(ce)
    val, vec = la.eig(cov)
    if clevel >= 1:
        space = vec[:, :int(clevel)]
    elif clevel < 1:
        val = val / val.sum()
        howmany = np.nonzero(np.cumsum(val) >= clevel)[0][0]
        space = vec[:, :howmany]
    return space.transpose()


def rand_space(ce, uc, clevel):
    inds = rand.ucevts(uc.shape[1], clevel, 2)
    return uc[:, inds].transpose()


DRMODES = {'istac': is_comp, 'ristac': is_comp_r, 'oldistac': is_comp_old, 'pca': pca_space, 'sig': siglength_comp,
           'rand': rand_space, 'ppca': ppca_space}


def compare(stim, evts1, evts2, length=LENGTH, lead=LEAD, compress='no', clevel=0, testprop=TESTPROP,
            bootstrap=BOOTSTRAP, report=None):
    '''
    build and compare models of event sequences self.evts[mi1] and
    self.evts[mi2]. The comparison is done by first building conditional
    ensembles of each event series with self.rcse.

    If istac is non-zero, it may be <float on (0,1)>. The ensembles will be
    projected into an iSTAC subspace that retains this proportion of the KL
    divergence before likelyhood testing. "istac" may also be the string
    "sig", in which case the ensembles will be reduced to the "significant"
    length, using sigLength.

    Likelyhood tests are performed bootstrap <int> times. In each test,
    both ensembles are randomly split into a training and testing set, with
    testprop of the samples in the testing set. The training sets are used
    to construct a gaussian model, and the testing sets are used to
    evaluate a log-likelyhood. All 4 test combinations are run (e.g.
    likelyhood that ensemble 1 test set came from the model trained on
    ensemble 1 training set, likelyhood that ensemble 2 test set came from
    the model trained on ensemble 1 training set, etc.). The results are
    organized in an <array[2,2,2]>. The first dimension of this array is
    which training set was used. The second dimension is which test set was
    used. The third dimension contains, in the first index, the mean of the
    set of bootstrap runs, and in the second index, the standard deviation
    of this set. Obviously, for "good" models, it is expected that the
    diagonal elements be greater (in mean) than the off-diagonal ones.

    '''
    rc1 = ece(stim, evts1, length, lead)
    rc2 = ece(stim, evts2, length, lead)
    if compress in DRMODES:
        uc = ucse(stim, 10000, length)
        ce = eqrc((rc1, rc2))
        cspace = DRMODES[compress](ce, uc, clevel)
        if report:
            report('Using %i components' % cspace.shape[0])
        rc1 = np.dot(cspace, rc1)
        rc2 = np.dot(cspace, rc2)
    return lltest(rc1, rc2, testprop, bootstrap)

