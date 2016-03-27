from mrjob.job import MRJob
from mrjob.util import cmd_line, bash_wrap
import os.path

class Grep(MRJob):

    def mapper_cmd(self):
        return bash_wrap('shell_cmd.sh')

if __name__ == '__main__':
    Grep.run()
