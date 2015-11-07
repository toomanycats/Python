######################################
# File Name : map_red_scape.py
# Author : Daniel Cuneo
# Creation Date : 11-07-2015
######################################
from mrjob.job import MRJob
from mrjob.step import MRStep
import indeed_scrape

ind = indeed_scrape.Indeed()
ind.query = "data science"
ind.build_api_string()

class MRWorker(MRJob):

    def mapper(self, _, line):
        zipcode, _,_,_,_,_,_= line.split(',')
        urls = ind.get_url(zipcode)
        for url in urls:
            if url is not None:
                yield url, None

    def reducer(self, url, _):
        skills = ind.parse_content(url)
        if skills is not None:
            yield skills, None

if __name__ == "__main__":
    MRWorker.run()
