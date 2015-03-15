import xml.etree.ElementTree as ET
from pyxnat import Interface
from os import path
import traceback
import re
import Tools
from operator import itemgetter

def parse_xml(phantom_QA_xml):    
    root = ET.fromstring(phantom_QA_xml)
    children = root.getchildren()
    
    pat = '.*prearchivePath.*'
    xml_path = regex_search(pat, children)
        
        
    dirs = xml_path.split('/')
    #first index is empty b/c of delimiter choice
    del(dirs[0])
    # URI="/fs/storage/XNAT/archive/ucsd_incoming/arc001/E-99999-P-9-20130329/RESOURCES/QA/QA_catalog.xml"
    index = dirs.index('XNAT')+1
    shortened_path = dirs[index:]
    path_prefix = ['/fs','ncanda-xnat']
    
    full_path_parts = path_prefix + shortened_path
    
    nii_path = path.normpath( '/'.join(full_path_parts) )    
    
    return nii_path
   
def get_stats(QA_xml_path):  
    ''' Get the children: snr, cnr, scale, nonlinear from:
<?xml version="1.0" encoding="utf-8"?>
<phantom>
<phantomType>MagphanEMR051</phantomType>
<snr>99.143401</snr>
<cnr>40.516546 47.209575 57.622379 52.288251</cnr>
<scale>0.998221 0.995950 0.994569</scale>
<nonlinear>0.439234 0.125164 0.201002</nonlinear>
<landmarkList coordinates="physical" space="RAS" count="165">
<fallbackCentroidCNR/>
<fallbackOrientationCNR/>
     '''
    
    # initialize with non-exist value 'O' for fall backs...if they are
    # present then this is overwritten
    out_put = {'fallbackCentroidCNR':' ', 'fallbackOrientationCNR':' '}
    
    try:
        tree = ET.parse(QA_xml_path)
        root = tree.getroot()
        children = root.getchildren()
        
        for child in children:
            key_pair  = parse_child(child)    
            for k,v in key_pair.iteritems():
                    out_put[k] = v

    except TypeError:
        print "Can't parse XML....Don't know what's going on here. \n"    
        tb = traceback.format_exc()
        print "Trace: %s" %tb
           
    return out_put

def parse_child(child):
    ''' Parse the XML child tags instead of relying upon list indices.
    Returns a tuple of a string, value, for use in a dict.'''

    if child.tag.lower() ==  'phantomtype':
        return {'phantom_type': child.text}

    elif child.tag.lower() == 'snr':
        return {'snr':child.text}

    elif child.tag.lower() == 'cnr':
        cnr = child.text.split()

        return {'cnr-1':cnr[0],'cnr-2':cnr[1], 'cnr-3':cnr[2], 'cnr-4':cnr[3]}

    elif child.tag.lower() == 'scale':
        scale = child.text.split()

        return {'scale-x':scale[0], 'scale-y':scale[1],'scale-z':scale[2]}

    elif child.tag.lower() == 'nonlinear':
        nl = child.text.split()

        return {'nonlin-x':nl[0], 'nonlin-y':nl[1], 'nonlin-z':nl[2]}

    elif child.tag.lower() == 'landmarklist':
        landmark_dict = child.attrib
        count = landmark_dict['count']

        return {'count':count}
        
    elif child.tag.lower() == 'fallbackcentroidcnr':
        return {'fallbackCentroidCNR':'X'}
    
    elif child.tag.lower() == 'fallbackorientationcnr':
        return {'fallbackOrientationCNR':'X'}

    else:
        raise Exception, "Failure to parse xml tag."
    
def write_header(row):

    header = ",".join(key for key in row)
    print header 
    
def main():
    keys = ('instit','scanner' ,'date', 'exp_key', 'snr', 'scale-x', 'scale-y','scale-z',
          'nonlin-x', 'nonlin-y', 'nonlin-z','fallbackCentroidCNR', 'fallbackOrientationCNR',
          'count')
    write_header(keys)
         
    pyx = Tools.PyXnatTools()
    
    phantom_keys = pyx.get_phantom_subject_keys()#'session_ID' in XNAT speak, and subject keys in our syntax
    
    columns = ['xnat:mrSessionData/PROJECT','xnat:mrSessionData/SCANNER_CSV','xnat:mrSessionData/INSERT_DATE',
               'xnat:mrSessionData/SESSION_ID']
    #returns a list of tuples..... [('ucsd_incoming', 'MR2OW1', '2013-04-07 20:02:05.576', 'NCANDA_E00171')]
    joined_columns_data = pyx.join_sub_keys_with_mr_session_keys(phantom_keys, columns)
    
    # need joined_col_data later for the instit, scanner_ID and date
    joined_data_dict = []
    for Tup in joined_columns_data:
        joined_data_dict.append({'instit':Tup[0], 'scanner':Tup[1], 'date':Tup[2][0:10], 'exp_key':Tup[3]})
    
    list_of_dicts = []
    for item in joined_data_dict:
        DL_path = path.join('/tmp', item['exp_key'])
        QA_xml_path = pyx.get_data_from_server(item['exp_key'], File='phantom.xml', dl_path=DL_path)
    
        stats = get_stats(QA_xml_path)
        
        #combine dicts into one dict, item is a single dict  in the list of dicts
        data = dict(item.items() + stats.items())
        list_of_dicts.append(data)
        
        #row = make_row(keys, item, stats)
        #write_row_to_stdout(row)    
    
    #sort_list_of_dicts(list_of_dicts)
    sorted_list = sorted(list_of_dicts, key=itemgetter('instit', 'scanner', 'date'))
    for dict_item in sorted_list:
        print ','.join([dict_item[k] for k in keys ])
    
if __name__ == "__main__":
    main()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    