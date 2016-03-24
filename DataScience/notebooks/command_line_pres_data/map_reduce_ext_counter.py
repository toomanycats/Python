from mrjob.job import MRJob
from mrjob.step import MRStep
from mrjob.util import bash_wrap

class Process(MRJob):

    def steps(self):
        return [
                MRStep(mapper_pre_filter = 'cut -d, -f 3',
                       mapper = self.mapper,
                       reducer = self.reducer
                       ),
                MRStep(mapper_cmd  = bash_wrap("sed 's/\"//g' | sed 's/\\\//g'")
                       )
                ]

    def mapper(self, _, line):
        yield line, 1

    def reducer(self, key, value):
        yield key, sum(value)


if __name__ == "__main__":
    Process().run()
