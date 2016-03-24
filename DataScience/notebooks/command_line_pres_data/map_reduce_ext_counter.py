######################################
# File Name : mr_chall.py
# Author : Daniel Cuneo
# Creation Date : 10-29-2015
######################################
from mrjob.job import MRJob

class Process(MRJob):

    def mapper_cmd(self):
       return 'cut -d, -f 3', 1

    def reducer(self, key, value):
        yield key, sum(value)

if __name__ == "__main__":
    Process().run()
