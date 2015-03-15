from os import path, remove
import re

import PyXnatTools


pyx = PyXnatTools.PyXnatTools()

keys = pyx.interface.select.experiments().get()

# example individual path B-00086-M-2-20130327.zip
basepath = "/fs/ncanda-share/burn2dvd"

out = []
count = 0 

for key in keys:
    date_archived = pyx.get_custom_variable(key, 'datetodvd')
    date_examined = pyx.get_custom_variable(key, 'findingsdate')
    if date_examined is not None:
        
        xml = pyx.get_xml_for_exp_id(key)
        reg = 'label="(?P<label>[A,B,C,D,E,X]-[0-9]{5}-[F,M,P,T]-[0-9]{1,2}-[0-9]{8})"'
        re_ob = re.search(reg, xml, flags=re.DOTALL)
        if re_ob:
            label = re_ob.group('label')
        
            full_path = path.join(basepath, date_archived, label + '.zip') 
            #print full_path
            if  path.exists(full_path):
                print full_path
                count += 1
             
#print count
            