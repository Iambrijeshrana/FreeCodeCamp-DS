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

sns.catplot(x="size", y="total_bill", kind="swarm",
            data=tips.query("size != 3"));

'''
The other option for choosing a default ordering is to take the levels of the 
category as they appear in the dataset. The ordering can also be controlled on a 
plot-specific basis using the order parameter.
'''
sns.catplot(x="smoker", y="tip", order=["No", "Yes"], data=tips);

# 2. Distributions of observations within categories
'''
As the size of the dataset grows, categorical scatter plots become limited in the 
information they can provide about the distribution of values within each category. 
When this happens, there are several approaches for summarizing the distributional 
information in ways that facilitate easy comparisons across the category levels.
'''
# i. Boxplot
sns.catplot(x="day", y="total_bill", kind="box", data=tips);
sns.catplot(x="day", y="total_bill", kind="boxen", data=tips);
# When adding a hue semantic, the box for each level of the semantic variable is 
# moved along the categorical axis so they don’t overlap:
# ii. Boxen
sns.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=tips);
sns.catplot(x="day", y="total_bill", hue="smoker", kind="boxen", data=tips);

'''
A related function, boxenplot(), draws a plot that is similar to a box plot but 
optimized for showing more information about the shape of the distribution. 
It is best suited for larger datasets:
'''
diamonds = sns.load_dataset("diamonds")
sns.catplot(x="color", y="price", kind="boxen",
            data=diamonds.sort_values("color"));    

# Combine boxen plot with swarm
g = sns.catplot(x="day", y="total_bill", kind="boxen", data=tips)
sns.swarmplot(x="day", y="total_bill", color="k", size=3, data=tips, ax=g.ax);

# iii. Violinplots
''' A different approach is a violinplot(), which combines a boxplot with the kernel density 
estimation procedure described in the distributions tutorial:
'''
sns.catplot(x="total_bill", y="day", hue="sex",
            kind="violin", data=tips);

# Combine Violin plot with swarm
g = sns.catplot(x="day", y="total_bill", kind="violin", data=tips)
sns.swarmplot(x="day", y="total_bill", color="k", size=3, data=tips, ax=g.ax);

# 3. Statistical estimation within categories
'''
For other applications, rather than showing the distribution within each category, you might 
want to show an estimate of the central tendency of the values. Seaborn has two main ways to 
show this information. Importantly, the basic API for these functions is identical to that 
for the ones discussed above.
'''
# i. Barplots
titanic = sns.load_dataset("titanic")
sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic);

# A special case for the bar plot is when you want to show the number of observations in 
# each category rather than computing a statistic for a second variable.
# it’s easy to do so with the countplot() function:

sns.catplot(x="deck", kind="count", palette="ch:.25", data=titanic);

sns.catplot(x="deck", kind="count", palette="pastel", hue='class', data=titanic);    

# ii. Point plots
# An alternative style for visualizing the same information is offered by the pointplot() function. 
sns.catplot(x="sex", y="survived", hue="class", kind="point", data=titanic);

sns.catplot(x="class", y="survived", hue="sex",
            palette={"male": "g", "female": "m"},
            markers=["^", "o"], linestyles=["-", "--"],
            kind="point", data=titanic)


''' To control the size and shape of plots made by the functions discussed above, you must set 
up the figure yourself using matplotlib commands: '''

f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="deck", data=titanic, color="c");


'''
Showing multiple relationships with facets
Just like relplot(), the fact that catplot() is built on a FacetGrid means that it is easy 
to add faceting variables to visualize higher-dimensional relationships:
'''

sns.catplot(x="day", y="total_bill", hue="smoker",
            col="time", aspect=.6,
            kind="swarm", data=tips);

g = sns.catplot(x="fare", y="survived", row="class",
                kind="box", orient="h", height=1.5, aspect=4,
                data=titanic.query("fare > 0"))
g.set(xscale="log");

' ************* Visualizing the distribution of a dataset ************* '

# When dealing with a set of data, often the first thing you’ll want to do is get a sense 
# for how the variables are distributed. 

# 1. Plotting univariate distributions

'''
The most convenient way to take a quick look at a univariate distribution in seaborn is 
the distplot() function. By default, this will draw a histogram and fit a 
kernel density estimate (KDE).
'''

x = np.random.normal(size=100)
sns.distplot(x);
sns.distplot(x, kde=False);
'''
Add bin size, also lets add a rug plot, which draws a small vertical tick at each observation. You 
can make the rug plot itself with the rugplot() function, but it is also available in distplot():
'''
sns.distplot(x, bins=20, kde=False, rug=True);
sns.distplot(x, bins=20, rug=True);

sns.distplot(x, hist=False, rug=True);

# 2. Plotting bivariate distributions
# let's visualize a bivariate distribution of two variables.

mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])

# The most familiar way to visualize a bivariate distribution is a scatterplot

sns.jointplot(x="x", y="y", data=df, kind='scatter');

# Hexbin plots - its good for large dataset

x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("white"):
    sns.jointplot(x=x, y=y, kind="hex", color="blue");

# Visualizing pairwise relationships in a dataset

'''
To plot multiple pairwise bivariate distributions in a dataset, you can use the 
pairplot() function. This creates a matrix of axes and shows the relationship 
for each pair of columns in a DataFrame. By default, it also draws the 
univariate distribution of each variable on the diagonal Axes:
'''    
# It will draw the figure only for numerical values
iris = sns.load_dataset("iris")
sns.pairplot(iris);    

sns.pairplot(tips)

'''
Specifying the hue parameter automatically changes the histograms to KDE plots 
to facilitate comparisons between multiple distributions.
'''
sns.pairplot(iris, hue="species");