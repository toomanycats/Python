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
    <h2>Enter key words</h2>
    <form action="." method="POST">
        <input type="text" name="text">
        <input type="submit" name="input_template" value="Send">
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

def plot_fig(kw, count):
    p = Bar(kw, count)
    p.title = "testing"
    p.xaxis.axis_label = "key word"
    p.yaxis.axis_label = "count"

    return p

@app.route('/')
def get_keywords():
    return input_template.render()

@app.route('/', methods=['POST'])
def keywords():
    return request.form['text']

@app.route('/')
def main():

    kws = keywords()
    kw, count = self.run_analysis(kws)

    p = plot_fig(kw, count)
    script, div = components(p)

    output_file(output_template)
    html = output_template.render(script=script, div=div)

    return encode_utf8(html)

def run_analysis(keywords):

    ind = indeed_scrape.Indeed()
    ind.query = keywords
    ind.num_samp = 0
    ind.add_loc = '^(94)'

    # test
    ind.handle_locations()
    ind.locations = ind.locations[0:10]

    ind.main()

    count, kw = ind.vectorizer(ind.df['summary_stem'])
    return kw, count

if __name__ == "__main__":
    app.debug = True
    app.run(port=5050)
