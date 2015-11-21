######################################
# File Name : web_version.py
# Author : Daniel Cuneo
# Creation Date : 11-21-2015
######################################
from flask import Flask
import indeed_scrape
import jinja2
from bokeh.embed import components
from bokeh.models.widgets import TextInput
from bokeh.io import output_file, show, vform
from bokeh.plotting import figure
from bokeh.util.string import encode_utf8

template = jinja2.Template("""
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

    <p> Enter the search keywords you use separated by a space. </p>

    {{ script }}

    {{ div }}

</body>

</html>
""")

app = Flask(__name__)

def plot_fig():
    p = figure(width=700, height=300)
    p.title = "testing"
    p.xaxis.axis_label = "key word"
    p.yaxis.axis_label = "count"
    p.line([1,2,3,4], [1,2,3,4])

    return p

@app.route('/')
def main():

    p = plot_fig()
    script, div = components(p)

    #output_file("web_test.html")
    #text_input = TextInput(value="default", title="Search terms:")
    #show(vform(text_input))
    html = template.render(script=script, div=div)

    return encode_utf8(html)

    #   ind = indeed_scrape.Indeed()
    #   ind.query = text_input
    #   ind.num_samp = 0
    #   ind.add_loc = '^(94)'
    #
    #   # test
    #   ind.handle_locations()
    #   ind.locations = ind.locations[0:10]
    #
    #   ind.main()
    #   df = ind.df

if __name__ == "__main__":
    app.debug = True
    app.run(port=5050)
