######################################
# File Name : web_version.py
# Author : Daniel Cuneo
# Creation Date : 11-21-2015
######################################
import logging
import pandas as pd
from flask import Flask
from flask import request, render_template, url_for
import indeed_scrape
import jinja2
from bokeh.embed import components
from bokeh.plotting import figure, output_file
from bokeh.util.string import encode_utf8
from bokeh.charts import Bar
import os

data_dir = os.getenv('OPENSHIFT_DATA_DIR')
logfile = os.path.join(data_dir, 'logfile.log')
logging.basicConfig(filename=logfile, level=logging.INFO)


input_template = jinja2.Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>indeed skill scraper</title>
    <meta charset="UTF-8">
</head>
<body>
    <h1>INDEED.COM JOB OPENINGS SKILL SCRAPER</h1>
    <form action="/main/" method="POST">
        Enter keywords you normally use to search for openings on indeed.com<br>
        <input type="text" name="kw"><br>
        Enter zipcodes<br>
        <input type="text" name="zipcodes"><br>
        <input type="submit" value="Submit" name="submit">
    </form>
</body>
</html>''')

output_template = jinja2.Template("""
<!DOCTYPE html>
<html lang="en-US">
<head>
    <title>indeed skill scraper results</title>
    <meta charset="UTF-8">
</head>

<link
    href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css"
    rel="stylesheet" type="text/css"
>
<script
    src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"
></script>

<body>

    <h1>INDEED.COM JOB OPENINGS SKILL SCRAPER RESULTS</h1>

    {{ script }}

    {{ div }}

</body>

</html>
""")

app = Flask(__name__)

def plot_fig(df, num):

    title_string = "Analysis of %i Postings" % num

    p = Bar(df, 'kw',
            values='count',
            title=title_string,
            title_text_font_size='15',
            color='blue',
            xlabel="keywords",
            ylabel="Count")

    return p

@app.route('/')
def get_keywords():
    return input_template.render()

@app.route('/main/', methods=['POST'])
def main():
    try:
        kws = request.form['kw']
        zips = request.form['zipcodes']
        logging.info(kws)
        logging.info(zips)

        kw, count, num, cities = run_analysis(kws, zips)

        df = pd.DataFrame(columns=['keywords','counts', 'cities'])

        df['kw'] = kw
        df['count'] = count
        df['cities'] = cities

        p = plot_fig(df, num)
        script, div = components(p)

        html = output_template.render(script=script, div=div)

        return encode_utf8(html)

    except Exception, err:
        logging.error(err)
        raise

def run_analysis(keywords, zipcodes):

    ind = indeed_scrape.Indeed()
    ind.query = keywords
    ind.stop_words = "and"
    ind.add_loc = zipcodes

    ind.main()
    df = ind.df
    df = df.drop_duplicates(['url']).dropna(how='any')

    count, kw = ind.vectorizer(df['summary_stem'])
    #convert from sparse matrix to single dim np array
    count = count.toarray().sum(axis=0)
    num = df['url'].count()

    return kw, count, num, df['city']

if __name__ == "__main__":
    app.debug = True
    app.run()
