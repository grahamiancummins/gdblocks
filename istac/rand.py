#!/usr/bin/env python
# encoding: utf-8
#Created by  on 2010-12-07.

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

import numpy as np
from gicdat.control import report

FROZEN = True
STIMSEED = 12345
UCSEED = 13579
JSEED = 2468
TSSEED = 121212


def evtjitter(jit, ns, cid):
	if FROZEN:
		rstate = np.random.get_state()
		np.random.seed(JSEED + 100*cid)
	j = np.round( np.random.normal(0.0, jit, ns) ).astype(np.int32)
	if FROZEN:
		np.random.set_state(rstate)
	return j

def stimulus(ns):
	if FROZEN:
		rstate = np.random.get_state()
		np.random.seed(STIMSEED)
	j= np.random.randn(ns)
	if FROZEN:
		np.random.set_state(rstate)
	return j

def ucevts(ns, size, eid=0):
	if FROZEN:
		rstate = np.random.get_state()
		np.random.seed(UCSEED + 100*eid)
	j= np.random.permutation(ns)[:size]
	if FROZEN:
		np.random.set_state(rstate)
	return j
	
def testset(ns, tid):
	if FROZEN:
		rstate = np.random.get_state()
		np.random.seed(TSSEED + 100*tid)
	j= np.random.permutation(ns)
	if FROZEN:
		np.random.set_state(rstate)
	return j
	
