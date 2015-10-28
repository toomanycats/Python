######################################
# File Name : practice_1.py
# Author : Daniel Cuneo
# Creation Date : 10-28-2015
######################################

from bokeh.sampledata.autompg import autompg as df
from bokeh.charts import Scatter, output_file, show

scatter = Scatter(df, x='mpg', y='hp', color='cyl', marker='origin',
                          title="mpg", xlabel="Miles Per Gallon", ylabel="Horsepower")

output_file('scatter.html')
show(scatter)
