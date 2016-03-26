from mrjob.job import MRJob
from mrjob.util import bash_wrap

class Grep(MRJob):

    def mapper_cmd(self):
      return bash_wrap("cut -d, -f 1 | pcregrep -no '(?sm)if\s*\(.*?\)' /dev/null | sed 's/\s//g'")


if __name__ == '__main__':
    Grep.run()
