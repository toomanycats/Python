######################################
# File Name : web_version.py
# Author : Daniel Cuneo
# Creation Date : 11-21-2015
######################################
import pandas as pd
from flask import Flask
from flask import request, render_template
import indeed_scrape
import jinja2
from bokeh.embed import components
from bokeh.plotting import figure, output_file
from bokeh.util.string import encode_utf8
from bokeh.charts import Bar

input_template = jinja2.Template('''
<!DOCTYPE html>
<html lang="en">
<body>
    <h1>INDEED.COM JOB OPENINGS SKILL SCRAPER</h1>
    <h2>Enter keywords you normally use to search for openings on indeed.com</h2>
    <form action="." method="POST">
        <div><input type="text" name="kw"></div>
    <h2>Enter zipcodes </h2>
        <div><input type="text" name="zipcodes"></div>
    <input type="submit" value="Submit">
    </form>
</body>
</html>''')

output_template = jinja2.Template("""
<!DOCTYPE html>
<html lang="en-US">

<link
    href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css"
    rel="stylesheet" type="text/css"
>
<script
    src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"
></script>

<body>

    <h1>INDEED.COM JOB OPENINGS SKILL SCRAPER</h1>

    {{ script }}

    {{ div }}

</body>

</html>
""")

app = Flask(__name__)

def plot_fig(df):

    title_string = "Analysis of %i Postings" % df['kw'].count()

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

@app.route('/', methods=['POST'])
def main():
    kws = request.form['kw']
    zips = request.form['zipcodes']
    kw, count = run_analysis(kws, zips)

    df = pd.DataFrame(columns=['keywords','counts'])

    df['kw'] = kw
    df['count'] = count

    p = plot_fig(df)
    script, div = components(p)

    html = output_template.render(script=script, div=div)

    return encode_utf8(html)

def run_analysis(keywords, zipcodes):

    ind = indeed_scrape.Indeed()
    ind.query = keywords
    ind.stop_words = "and"
    ind.add_loc = zipcodes
    ind.locations = ind.handle_locations()

    ind.main()
    df = ind.df
    df = df.drop_duplicates(['url']).dropna(how='any')

    count, kw = ind.vectorizer(df['summary_stem'])
    #convert from sparse matrix to single dim np array
    count = count.toarray().sum(axis=0)

    return kw, count

if __name__ == "__main__":
    app.debug = True
    app.run(port=5050)
