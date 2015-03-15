import PySQL
import GroupAnalysisTools

pysql = PySQL.PySQL()
writer = GroupAnalysisTools.WriteCovCsv('/fs/u00/dpc/all_ncanda_cov.txt')

sub = pysql.get_complete_sub_list()

for s in sub:
    try:
        print s
        cov = GroupAnalysisTools.NcandaGroupCov(s)
        cov_dict = cov.get_cov()
        writer.write_to_csv(cov_dict)
    except:
        pass    

writer.close_file_object()                                  