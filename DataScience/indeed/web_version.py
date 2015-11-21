######################################
# File Name : web_version.py
# Author : Daniel Cuneo
# Creation Date : 11-21-2015
######################################
import indeed_scrape
import jinja2
from bokeh.embed import components
from bokeh.models.widgets import TextInput
from bokeh.io import output_file, show, vform
from bokeh.plotting import figure

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

output_file("text_input.html")
text_input = TextInput(value="default", title="Search terms:")
template.render(script=script, div=div)
show(vform(text_input))

ind = indeed_scrape.Indeed()
ind.query = text_input
ind.num_samp = 0
ind.add_loc = '^(94)'

ind.main()
df = ind.df

