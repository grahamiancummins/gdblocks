import numpy as n

from gicdat.cext.tests import buildnload

pf = buildnload('portforsgic')
print pf.__file__
print dir(pf)

tem=n.random.uniform(-1, 1, 30).astype(n.float32)
var=n.random.uniform(-1, 1, 30).astype(n.float32)

print
print 'spike dist'
pf.spikeDistance(tem, var, 0)
print "ok"

print
print 'transform'
e2=n.array([2,4,10,23,24,25])
e1=n.array([1,3,5,9,15,21,25,30])
l=pf.evttransform(e1,e2,.5)
print e1
print e2
print l
print "ok"


print
print 'mi_direct'
e1 = n.random.randint(0, 20, 50)
e2=n.random.randint(0, 10, 50)
l=pf.mi_direct(e1,e2)
print e1
print e2
print l
#l=pf.mi_direct(e1,e1)
#import gdblocks.portfors.mi as pmi
#print "reference"
#l2 = pmi.minf_direct({'stims':e1, 'evts':e2})
#print l2
#l2 = pmi.minf_direct({'stims':e1, 'evts':e1})
#print l2
print "ok"