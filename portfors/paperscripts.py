#!/usr/bin/env python -3
# encoding: utf-8

#Created by Graham Cummins on Jun 7, 2011

# Copyright (C) 2011 Graham I Cummins
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later 
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA 02111-1307 USA
#

import timingUI as ti
import selectUI as si

def allstims(cdoc):
	stims = set()
	for k in cdoc.keys(0, 'cell'):
		d = cdoc[k]['stimclasses']
		l = [d[s]['file'] for s in d]
		stims = stims.union(l)
	return stims
