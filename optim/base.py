## Copyright (C) 2005-2006 Graham I Cummins
## This program is free software; you can redistribute it and/or modify it under 
## the terms of the GNU General Public License as published by the Free Software 
## Foundation; either version 2 of the License, or (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but WITHOUT ANY 
## WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
## PARTICULAR PURPOSE. See the GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License along with 
## this program; if not, write to the Free Software Foundation, Inc., 59 Temple 
## Place, Suite 330, Boston, MA 02111-1307 USA
## 
from gicdat.control import report, error
from gicdat.stdblocks.gicparse import GicLogParse
import numpy as np
import time, threading
import os, sys


class Model(object):
    '''
    SUBCLASS THIS

    Attributes:

    perfect: if specified, this is the best possible fitness value that could
    be returned. For example in algorithms where fitness is -1*error, for some
    error measure, perfect=0.

    '''
    perfect = None

    def __init__(self, myparams):
        self.initpars = myparams

    def eval(self, pars):
        '''
        evaluate the parameter set pars <N-tuple(float)>, and return a tuple
        (fit <float>, evalcond <M-tuple(float)>, which contains the value of
        the modeled function at the parameters, and any special evaluation
        conditions that were used in this evaluation (only use these if there
        is site-specific or stochastic evaluation. Otherwise, return an empty
        tuple here)

        '''
        f = np.multiply.reduce(pars)
        return (f, ())

    def toNode(self):
        d = {'tag': 'optim.model',
             '_class': str(self.__class__),
             '_hash': hash(self.__class__)}
        d.update(self.initpars)
        return Node(d, None, None)


class Alg(object):
    '''
    SUBCLASS THIS

    Attribute range - This will be a ParamRange instance

    Algs should assume that only the first self.range.nd of the parameters in
    a candidate unit are controlled by the search (the remaining elements are
    stochastic or site-specific evaluation conditions), and should always
    return candidate parameter sets of length self.range.nd (and that
    fit within the range's bounds)

    '''

    def __init__(self, pars, prange):
        self.pars = pars
        self.range = prange


    def arrays(self, units):
        '''
        Converts a "units" dictionary <{N-tuple(float):float}> to numpy arrays.
        Returns (array((M,1), float), array((M,L), float)), where the first array is the
        fitness values (the set of values from the dictionary), and the MxL array is the
        set of all parameters (each row is one key tuple from the dictionary). L==N if
        not self.nd, otherwise L=self.nd (and each row is the first nd elements of the
        associated tuple)

        '''
        f = []
        v = []
        for i in units.items():
            f.append(i[1])
            v.append(i[0])
        f = np.array(f)
        v = np.array(v)
        v = v[:, :self.range.nd]
        return (f, v)

    def initcond(self):
        '''
        Return a parameter set representing an initial condition (with no previous
        evaluations to use in deciding on it). This base class returns a random
        parameter set.

        '''
        return self.range.random()


    def next(self, units):
        '''
        given the unit population "units" <{N-tuple(float):float}>, suggest the
        next parameter set to try. Return value is a parameter set
        <M-tuple(float)>. M == self.nd if nd is set, otherwise M==N.

        '''
        return self.range.random()


    def kill(self, units):
        '''
        delete one of the units
        '''
        u = units.keys()[0]
        del (units[u])

    def toNode(self):
        d = {'tag': 'optim.alg',
             '_class': str(self.__class__),
             '_hash': hash(self.__class__)}
        d.update(self.pars)
        return Node(d, None, None)


class ParamRange(object):
    def __init__(self, a):
        '''
        a is <array((N,3), real)>, which each row specifying a parameter range
        using (minimum, maximum, number of steps). Both bounds are inclusive

        '''
        self.min = a[:, 0]
        self.max = a[:, 1]
        self.shape = a[:, 2]
        self.nd = a.shape[0]
        self.step = (self.max.astype(np.float64) - self.min) / self.shape

    def fromInt16(self, p):
        '''
        map p <N-tuple(int)> onto and <array((N,), float)> that contains
        parameters in the described range, by mapping the range of an int16
        onto the indicated ranges.

        '''
        p = (p.astype(np.float64) + 32768) / 65536.
        p = np.around(p * self.shape).astype(np.int32)
        return self.min + p * self.step


    def fromFrac(self, p):
        p = np.around(p * self.shape)
        return self.min + p * self.step

    def random(self):
        '''
        return a random set of parameter values as a 1D array. (Although the
        values are random, they are within the allowed ranges and precisions)

        '''
        q = np.random.uniform(0, 1, self.nd)
        q = self.fromFrac(q)
        return tuple(q)

    def toNode(self):
        d = {'tag': 'optim.paramrange',
             '_class': str(self.__class__),
             '_hash': hash(self.__class__)}
        dat = np.column_stack([self.min, self.max, self.shape])
        return Node(d, None, dat)


class Store(object):
    def __init__(self, fname):
        '''
        Create a store bound to the file fname.
        '''
        self.parse = LogParse()
        self.parse.attach(fname)
        self.bn = 'u'
        if not self.bn in self.parse.names:
            self.parse.names[self.bn] = 1

    def order(self, x, y):
        return cmp(int(x[len(self.bn):]), int(y[len(self.bn):]))

    def get(self, n):
        '''
        return a  {M-tuple(float):float} with n elements, containing the
        mapping of parameter tuples to fitness for the n most recent units. If
        n is 0, return the whole record

        '''
        ns = [name for name in self.parse.nodes() if name.startswith(self.bn)]
        if n:
            ns = sorted(ns, self.order)
            ns = ns[-n:]
        units = {}
        zf = self.parse.zfile()
        for name in ns:
            node = self.parse.readNode(name, zf)
            units[node['pars']] = node['fit']
        zf.close()
        return units

    def size(self):
        '''return the number of stored units <int>'''
        return self.parse.names[self.bn] - 1

    def record(self, fit, ps):
        '''
        add the unit with parameter set ps <N-tuple(float)> and fitness fit
        <float> to storage

        '''
        n = Node({'fit': fit, 'pars': ps})
        self.parse.add(n, self.bn)

    def saveNodes(self, d):
        '''store a dict d of nodes with explicit names'''
        for k in d:
            self.parse.writeNode(k, d[k])


class Optimizer(object):
    def __init__(self, pars, model, alg, store):
        '''
        Pars is a dictionary containing parameters, including:
            threads <int>: max number of threads
            size <int>: size of the population
            time <float>: maximum time to run for in hours
        '''
        self.pars = pars
        self.alg = alg
        self.model = model
        self.store = store
        self.prep()

    @property
    def range(self):
        return self.alg.range

    def test(self):
        '''
        Generate a random parameter set and evaluate it.  Returns a tuple
        (float, tuple). The tuple is the random set of parameters, with any evaluation conditions specified by the model appended. The float is
        the fitness

        '''
        rc = self.range.random()
        fit, ec = self.model.eval(rc)
        return (fit, rc + ec)

    def fitstats(self, s=False):
        '''
        Return a tuple (max, min, mean, std) of the current population fitness
        values. If s is True, return a string describing these stats instead.

        '''
        f = np.array([i[1] for i in self.units.items()])
        stats = (f.max(), f.min(), f.mean(), f.std())
        if s:
            return "best:%.4g, worst:%.4g, mean:%.4g, std:%.4g" % stats
        else:
            return stats

    def describe(self):
        n = {'tag': 'optim.base',
             '_class': str(self.__class__),
             '_hash': hash(self.__class__),
             '_timeofrun': tuple(time.localtime())}
        n.update(self.pars)
        d = {'optim.base': Node(n, None, None),
             'optim.alg': self.alg.toNode(),
             'optim.searchrange': self.alg.range.toNode(),
             'optim.model': self.model.toNode()}
        return d

    def prep(self):
        '''
        Set up the optimization run.

        '''
        self.abort = False
        self.best = (None, None)
        self.nunits = self.store.size()
        self.threads = []
        if self.nunits:
            self.units = self.store.get(self.pars['size'])
            z = self.units.items()
            snd = len(z[0][0])
            if snd < self.range.nd:
                raise StandardError(
                    'Attempt to resume %i-d optimizer from a storage of %i-d parameters' % (self.range.nd, snd))
            elif snd > self.range.nd:
                report(
                    'resuming an %i-d optimizer from a store of %i-d parameter sets. Hopefully this is intentional (eg. there are eval-conditions in the model). If not, dont run this optimizer' % (
                    self.range.nd, snd))
            f = np.array([i[1] for i in z])
            bi = f.argmax()
            self.best = (f[bi], z[bi][0])
        #FIXME - sanity check for the stored components should go here
        else:
            self.store.saveNodes(self.describe())
            self.units = {}
        report("prep complete")
        self.lock = threading.Lock()
        if self.units:
            report("Resuming. %i units (%s)" % (self.nunits, self.fitstats(True)))


    def checkthreads(self):
        '''
        return True if there are "free threads" and False otherwise. There are
        "free threads" if the number of threads in self.threads which are alive
        is less than self.pars.get('threads', 0). As a side-effect of the
        check, any threads in self.threads which are not alive are removed.

        '''
        nt = self.pars.get('threads', 0)
        if len(self.threads) < nt:
            return True
        th = self.threads[:]
        for t in th:
            if not t.isAlive():
                self.threads.remove(t)
        if len(self.threads) < nt:
            return True
        else:
            return False

    def threadcall(self, method, args):
        '''
        If len(self.threads) is less than self.par.get('threads', 0), create a
        new thread, add it to self.threads, and run method, args in it.

        If 'threads' is 0, run the method locally.

        If threads is not 0, but len(self.threads) >= threads, blocks until a
        thread opens up (calls self.checkthreads once a second to clean up
        finished threads)

        '''
        nt = self.pars.get('threads', 0)
        if not nt:
            apply(method, args)
            return
        if nt <= len(self.threads):
            self.checkthreads()
        while not self.checkthreads():
            self.threads[0].join(1.0)
        t = threading.Thread(target=method, args=args)
        t.start()
        self.threads.append(t)


    def lockcall(self, method, args):
        '''
        Attempt to acquire self.lock, execute the indicated method with the
        indicated arguments, and release the lock. The method call is wrapped
        in a general try/except, gauranteeing that the lock is always released
        (but making failed method calls hard to debug)

        '''
        self.lock.acquire()
        try:
            apply(method, args)
        except:
            raise
            report("Call to %s%s failed" % (str(method), str(args)))
        self.lock.release()


    def run(self, background=False):
        '''
        Run the optimizer. If background is True, run in a thread.
        and return the thread.
        '''
        if background:
            t = threading.Thread(target=self.run, args=(False,))
            t.start()
            return t
        self.start = time.time()
        self.abort = False
        report("Starting Run (pid=%s)" % (str(os.getpid(), )))
        while not self.done():
            try:
                self.threadcall(self.search, ())
                if self.nunits and not self.nunits % 200:
                    report("Evaluate %i. %s" % (self.nunits, self.fitstats(True)))
            except KeyboardInterrupt:
                report('Manual abort. Shutting down.')
                self.abort = True
        report("Run Completed")

    def search(self):
        '''
        This is the main loop called by run (sometimes in a thread). It calls
        self.alg.next(self.units, self.range.nd) to get a paremeter set, calls
        self.model.eval to evaluate it, and calls self.insert to store it.

        '''
        if len(self.units) < self.pars['size']:
            ps = self.alg.initcond()
        else:
            ps = self.alg.next(self.units)
        try:
            (fit, ec) = self.model.eval(ps)
        except Exception:
            e = sys.exc_info()
            error(e, None, rep="GA Eval error")
            fit = "model.eval threw exception"
        if type(fit) in [str, unicode]:
            report('Eval error, %s -> %s' % (str(ps), fit))
        else:
            self.lockcall(self.insert, (fit, ps + ec))

    def insert(self, fit, ps):
        '''
        Add a unit to the population and store it in self.store

        '''
        if len(self.units) >= self.pars['size']:
            self.alg.kill(self.units)
        self.units[ps] = fit
        self.store.record(fit, ps)
        self.nunits += 1
        if fit > self.best[0]:
            sps = ''
            for p in ps:
                sps += "%.4g, " % p
            sps = sps[:-2]
            report('new best %.4f  (%s)' % (fit, sps))
            self.best = (fit, ps)
            if self.model.perfect != None and fit == self.model.perfect:
                report(
                    "You hear the singing of Angels and a chorus of trumpets, hailing the birth of a perfect unit. All further competition is futile. Exiting")
                self.abort = "perfected"


    def runtime(self):
        try:
            return (time.time() - self.start) / 3600.0
        except:
            return 0

    def done(self):
        '''return True if the conditions for completing a run are met.'''
        if self.abort:
            return True
        q = self.pars.get('time')
        if q and self.runtime() >= q:
            return True
        return False


