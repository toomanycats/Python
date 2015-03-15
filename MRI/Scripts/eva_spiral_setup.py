import NcandaSpiral
from optparse import OptionParser
import LoggingTools
from os import path, mkdir

parser = OptionParser()

parser.add_option("-s", "--subject-label", dest="sub_label", type=str, default=None,
                  help="Human readable subject label from XNAT: fs/ncanda-xnat/archive/sri_incoming/arc001/<sub_label>.")

parser.add_option("-L", "--log-level", dest="log_level", type=str, default='info',
                  help="Log levels are: 'debug', 'info', 'error' ")

(options, args) = parser.parse_args()

sandbox_path = NcandaSpiral.sandbox_path
base = path.join(sandbox_path, options.sub_label)

if not path.exists(base):
    mkdir(base)

log_path = path.join(base, 'logfile.txt')

log = LoggingTools.SetupLogger(log_path, options.log_level).get_module_logger()

spiral = NcandaSpiral.SpiralTask(options.sub_label)
spiral.eva_truncated_version()
