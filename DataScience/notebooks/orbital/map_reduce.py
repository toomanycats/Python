from mrjob.job import MRJob
from mrjob.util import bash_wrap

class ShellCmds(MRJob):
    def mapper_cmd(self):
        return bash_wrap('cut -d, -f 1 | xargs -n 1 grep -i "copyright" | sed "s/\W//g" | sed "s/[0-9]//g"')

    def reducer_cmd(self):
       return bash_wrap('sort | uniq -c | sort -n')

if __name__ == '__main__':
    ShellCmds.run()
