#####################################
# File Name : map_red_scape.py
# Author : Daniel Cuneo
# Creation Date : 11-07-2015
######################################
from mrjob.job import MRJob
from mrjob.step import MRStep
import indeed_scrape

mrj = MRJob()

class MRWorker(MRJob):

    def configure_options(self):
        super(MRWorker, self).configure_options()
        self.add_file_option('--tokens')

    def url_mapper_init(self):
        self.ind = indeed_scrape.Indeed()
        self.ind.query = "data science"
        self.ind.config_path = "tokens.cfg"
        self.ind.load_config()
        self.ind.build_api_string()

    def url_mapper(self, _, line):
        zipcode, _,_,_,_,_,_= line.split(',')
        urls = self.ind.get_url(zipcode)
        for url, city in urls:
            yield city, url

    def cont_mapper_init(self):
        self.ind = indeed_scrape.Indeed()

    def cont_mapper(self, city, url):
        #mrj.set_status("reducer working")
        content = self.ind.get_content(url)
        yield city, content

    def parse(self, city, content):
        skills = self.ind.parse_content(content)
        yield city, skills

    def steps(self):
        return [MRStep(mapper=self.url_mapper, mapper_init=self.url_mapper_init),
                MRStep(mapper=self.cont_mapper, mapper_init=self.cont_mapper_init),
                MRStep(mapper=self.parse, mapper_init=self.cont_mapper_init)
               ]


if __name__ == "__main__":
    MRWorker.run()
