# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:47:55 2020

@author: Brijesh.R
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


# To show the relationship between two varianbles via Scatter plot and line chart

# Scatter plot ( Scatter plot is use to show the relationship between two or more categorical variavbles)
tips = sns.load_dataset("tips")
sns.relplot(x="total_bill", y="tip", data=tips);

#sns.scatterplot(x="total_bill", y="tip", data=tips)

# lets add one more variable i.e. thired variable using HUE
sns.relplot(x="total_bill", y="tip", data=tips, hue='smoker');
# To add the style in soker (in case we want want to visualize smokers diffrently)
sns.relplot(x="total_bill", y="tip", data=tips, hue='smoker', style='smoker');

# we can add 4th variable also using style like - 
sns.relplot(x="total_bill", y="tip", data=tips, hue='smoker', style='time');

# In the above graph things aee not so clear so lets draw two graph based on time 
# case if we wanr to see the visualization in two grphs

sns.relplot(x="total_bill", y="tip", hue="smoker",
            col="time", data=tips);
'''
In the examples above, the hue semantic was categorical, so the default qualitative palette was applied. 
If the hue semantic is numeric (specifically, if it can be cast to float), the default coloring switches 
to a sequential palette:
'''
sns.relplot(x="total_bill", y="tip", hue="size", data=tips);

# Now lets change the size of dots
sns.relplot(x="total_bill", y="tip", size="smoker", data=tips);


''' 
Now lets think you want to see the change in the variables overt the periods of time.
Here time is an important for us. In such situations scatter plot is not a good chice 
And insted of scatter plot we will use Line Charts
'''
# Line Chart

df = pd.DataFrame(dict(time=np.arange(500),
                       value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df, markers=True)
g.fig.autofmt_xdate()


'''
Aggregation and representing uncertainty
More complex datasets will have multiple measurements for the same value of the x variable. The default behavior in seaborn is to aggregate the multiple measurements at each x value by plotting the mean and the 95% confidence interval around the mean:
'''
fmri = sns.load_dataset("fmri")
sns.relplot(x="timepoint", y="signal", kind="line", data=fmri, markers=True, dashes=False);

# Remove confidance interval 
sns.relplot(x="timepoint", y="signal", ci=None, kind="line", data=fmri);

'''
Another good option, especially with larger data, is to represent the spread of the distribution at each 
timepoint by plotting the standard deviation instead of a confidence interval:
'''
sns.relplot(x="timepoint", y="signal", kind="line", ci="sd", data=fmri);

# Add more variable as we did for scatter plot 
sns.relplot(x="timepoint", y="signal", hue="event", kind="line", data=fmri);


sns.relplot(x="timepoint", y="signal", hue="region", style="event",
            dashes=False, markers=True, kind="line", data=fmri);


# Ploting data with Date

df = pd.DataFrame(dict(time=pd.date_range("2017-1-1", periods=500),
                       value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)
# In the above graph dates are overlapping each other so lets solve this problem
g.fig.autofmt_xdate()

'''
# See the diffrence
sns.relplot(x="total_bill", y="tip", hue="smoker",
            style="time", data=tips);

sns.relplot(x="total_bill", y="tip", hue="smoker",
            col="time", data=tips);
'''

'''facet mean - a particular aspect or feature of something. or Paksh
You can also show the influence two variables this way: one by faceting on the columns and one by 
faceting on the rows. As you start adding more variables to the grid, you may want to decrease the 
figure size. Remember that the size FacetGrid is parameterized by the height and aspect 
ratio of each facet:
'''    
sns.relplot(x="timepoint", y="signal", hue="subject",
            col="region", row="event", height=4,
            kind="line", estimator=None, data=fmri);

'''
When you want to examine effects across many levels of a variable, 
it can be a good idea to facet that variable on the columns and then 
“wrap” the facets into the rows:
'''

sns.relplot(x="timepoint", y="signal", hue="event", style="event",
            col="subject", col_wrap=5,
            height=3, aspect=.75, linewidth=2.5,
            kind="line", data=fmri.query("region == 'frontal'"))

'''
********** Plotting with categorical data ********** 
'''

# 1. Categorical scatterplots
tips.head()
sns.catplot(x="day", y="total_bill", data=tips);

sns.catplot(y="day", x="total_bill", data=tips);

#The jitter parameter controls the magnitude of jitter or disables it altogether:

sns.catplot(x="day", y="total_bill", jitter=False, data=tips);

'''
Lets adjusts the points along the categorical axis using an algorithm that prevents 
them from overlapping. It can give a better representation of the distribution of 
observations, although it only works well for relatively small datasets. 
This kind of plot is sometimes called a “beeswarm” and is drawn in seaborn by 
swarmplot(), which is activated by setting kind="swarm" in catplot():
'''    

sns.catplot(x="day", y="total_bill", kind="swarm", data=tips);

'''
Similar to the relational plots, it’s possible to add another dimension to a 
categorical plot by using a hue semantic. (The categorical plots do not currently 
                                           support size or style semantics). 
Each different categorical plotting function handles the hue semantic differently. 
For the scatter plots, it is only necessary to change the color of the points:
'''    
sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips);

# categorical plotting functions try to infer the order of categories from the data.