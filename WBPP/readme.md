# World Bank Poverty Prediction Problem Forked
### Background
In order to eliminate poverty, it is imperative to be able to identify households suffering from poverty and target them with assistance. However, the identification of households in poverty relies on data from consumption surveys that is difficult, expensive, and time-consuming to collect.

Therefore, recent efforts have been focused on the use of “rapid surveys” that rely a limited number of poverty identifiers that serve as effective proxies for the calculation of a household’s poverty status.

### Objectives
The World Bank has asked you to identify the most important variables that determine a household’s poverty status to help them reduce the cost associated with compiling data to predict poverty.
### Data
The data provided for analysis is household responses to a World bank consumption survey. Each observation has a unique household id to reflect the survey responses of that distinct household. Further, each household is labeled in or out of poverty through the Poor indicator variable. 

Notice that all of the variables are encoded as random character strings but reflect actual survey questions. For categorical variables, these variables may reflect questions such as does your household have items such as Bar soap, Cooking oil, Matches, and Salt. Numeric questions often ask things like How many working cell phones in total does your household own? or How many separate rooms do the members of your household occupy? The project is not meant for you to determine the real meaning of the variables you select, rather just identify the best variables in their encoded state to best predict poverty.

Two datasets in the format pictured above are supplied, one for model training and one for model testing. No external data beyond what is provided should be used for modeling.
### Error Metric
When evaluating your model’s performance in its ability to predict a household’s poverty status, you should use the logloss error metric. We define the logloss metric through the following formula:

![equation](https://latex.codecogs.com/gif.latex?\fn_cm&space;\frac{1}{N}\sum^{N}_{i=1}[y_ilog(\hat{y}_i)&plus;(1-y_i)log(1-\hat{y}_i)])

The logloss metric any value from 0 to positive infinity in which a model scoring a 0 is a perfect classifier. Also, notice how the logloss error function operates. The metric rewards a model that confidently classifies a household correctly and punishes a model that is overconfident for wrong classifications. For example, a model that predicts a high probability of a household being poor and the household is actually poor will receive a lower logloss score than a model that predicts a high probability of poverty for a household that is not poor.

This project has been adapted from a competition on the data science website: DrivenData.
