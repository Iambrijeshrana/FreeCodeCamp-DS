Time series Dataset that we are using to predict fututre values - India_Key_Commodities_Retail_Prices_1997_2015.csv

It's a grouped time series dataset and for this problem we are using Facebook Prophet model to forecast future values.

Steps to predict future values -

1. Import the csv file into MS SQL database
2. Read the  data from database and store into dataframe
3. Apply Facebook Prophet model to predict the values for next 36 months
4. Store the result in MS SQL database 