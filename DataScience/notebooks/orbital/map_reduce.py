from mrjob.job import MRJob
from mrjob.util import bash_wrap

class MRWordFrequencyCount(MRJob):

    def mapper(self, _, line):
        yield line.split(',')[0]

    def reducer(self, _, line):
        return bash_wrap("pcregrep -no '(?sm)if\s*\(.*?\)' /dev/null | sed 's/\s//g'")

if __name__ == '__main__':
    MRWordFrequencyCount.run()
