######################################
# File Name : map_red_scape.py
# Author : Daniel Cuneo
# Creation Date : 11-07-2015
######################################
from mrjob.job import MRJob
#from mrjob.step import MRStep
import indeed_scrape


class MRWorker(MRJob):

    def configure_options(self):
        super(MRWorker, self).configure_options()
        self.add_file_option('--tokens')

    def mapper_init(self):
        self.ind = indeed_scrape.Indeed()
        self.ind.query = "data science"
        self.ind.config_path = "tokens.cfg"
        self.ind.load_config()
        self.ind.build_api_string()

    def mapper(self, _, line):
        zipcode, _,_,_,_,_,_= line.split(',')
        urls = self.ind.get_url(zipcode)
        for url in urls:
            if url is not None:
                yield url, None
    def reducer_init(self):
        self.ind = indeed_scrape.Indeed()

    def reducer(self, url, _):
        skills = self.ind.parse_content(url)
        if skills is not None:
            yield skills, None

if __name__ == "__main__":
    MRWorker.run()
