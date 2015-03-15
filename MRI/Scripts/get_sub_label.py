#!/usr/bin/python
import sys
import PyXnatTools

if len(sys.argv) < 1:
    print "Usage: <subject ID>, returns site code, i.e. 'sri', and subject label."

sub_id = sys.argv[1]    
pyx = PyXnatTools.PyXnatTools()

site, label = pyx.get_site_from_sub_ID(sub_id)
print "%s:%s:%s" %(sub_id, site, label)

        