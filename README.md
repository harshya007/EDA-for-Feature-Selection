# EDA-for-Feature-Selection
Exploratory Data Analysis (EDA) is an approach in data analysis that uses various techniques to maximize insight into a dataset, uncover underlying structure, analyze relationships between variables, detect outliers and anomalies, test underlying assumptions, and feature selection for training Machine Learning models. So, if you want to learn how to choose the best features for training Machine Learning models using EDA, this article is for you. In this article, I’ll take you through the task of EDA for Feature Selection using Python.


EDA for Feature Selection: Process We Can Follow
EDA is a valuable step in a data science workflow, particularly for feature selection. It helps you know about the best candidates for features based on their relationship with the target variable and each other, their relevance, and their predictive power.

Below is the process you can follow while performing EDA for Feature Selection:

Begin by getting familiar with the dataset. It includes understanding the size, scope, and nature of the data (structured vs. unstructured), identifying each feature and its type (numerical, categorical), and recognizing the target variable.
Identify and handle missing values by imputation, deletion, or estimation, depending on their nature and the proportion of missing data.
Analyze the distribution of each feature using histograms, density plots, or bar charts for categorical variables.
Conduct correlation analysis using Pearson, Spearman, or Kendall correlation coefficients to assess the relationship between numeric features and the target variable.
Utilize scatter plots, pair plots, and heat maps to explore relationships between features.
Based on insights gained through EDA, manually remove redundant features, features with very little variance, or features highly correlated with others.
To get started with EDA for feature selection, we need a dataset. I found an ideal dataset for this task. You can download it from here.


EDA for Feature Selection using Python
Now, let’s get started with EDA for Feature Selection by importing the necessary Python libraries and the dataset:

import pandas as pd

data = pd.read_csv("dynamic_pricing.csv")

print(data.head())
1
import pandas as pd
2
​
3
data = pd.read_csv("dynamic_pricing.csv")
4
​
5
print(data.head())
   Number_of_Riders  Number_of_Drivers Location_Category  \
0                90                 45             Urban   
1                58                 39          Suburban   
2                42                 31             Rural   
3                89                 28             Rural   
4                78                 22             Rural   

  Customer_Loyalty_Status  Number_of_Past_Rides  Average_Ratings  \
0                  Silver                    13             4.47   
1                  Silver                    72             4.06   
2                  Silver                     0             3.99   
3                 Regular                    67             4.31   
4                 Regular                    74             3.77   

  Time_of_Booking Vehicle_Type  Expected_Ride_Duration  \
0           Night      Premium                      90   
1         Evening      Economy                      43   
2       Afternoon      Premium                      76   
3       Afternoon      Premium                     134   
4       Afternoon      Economy                     149   

   Historical_Cost_of_Ride  
0               284.257273  
1               173.874753  
2               329.795469  
3               470.201232  
4               579.681422  
Our goal is to identify the most important features for predicting the Historical_Cost_of_Ride. Let’s start with the data quality check to identify any missing values or incorrect data types:

# check for missing values and data types
missing_values = data.isnull().sum()
data_types = data.dtypes

missing_values_report = pd.DataFrame({'Missing Values': missing_values, 'Data Type': data_types})
print(missing_values_report)
1
# check for missing values and data types
2
missing_values = data.isnull().sum()
3
data_types = data.dtypes
4
​
5
missing_values_report = pd.DataFrame({'Missing Values': missing_values, 'Data Type': data_types})
6
print(missing_values_report)
                         Missing Values Data Type
Number_of_Riders                      0     int64
Number_of_Drivers                     0     int64
Location_Category                     0    object
Customer_Loyalty_Status               0    object
Number_of_Past_Rides                  0     int64
Average_Ratings                       0   float64
Time_of_Booking                       0    object
Vehicle_Type                          0    object
Expected_Ride_Duration                0     int64
Historical_Cost_of_Ride               0   float64
The dataset does not contain any missing values, and the data types appear appropriate for each feature.

Univariate Analysis
With data quality checks completed and no immediate data cleaning required, we can move on to Univariate Analysis. It will help us understand the distribution of each variable individually. I’ll start by analyzing the distribution of numerical features to understand their distribution:


axes[i//2, i%2].set_xlabel('')
    axes[i//2, i%2].set_ylabel('')

print(descriptive_stats)
1
import matplotlib.pyplot as plt
2
import seaborn as sns
3
​
4
# set the aesthetics for the plots
5
sns.set_style("whitegrid")
6
​
7
# define the numerical and categorical columns
8
numerical_cols = ['Number_of_Riders', 'Number_of_Drivers', 'Number_of_Past_Rides',
9
                  'Average_Ratings', 'Expected_Ride_Duration', 'Historical_Cost_of_Ride']
10
categorical_cols = ['Location_Category', 'Customer_Loyalty_Status', 'Time_of_Booking', 'Vehicle_Type']
11
​
12
# descriptive statistics for numerical features
13
descriptive_stats = data[numerical_cols].describe()
14
​
15
# plotting distributions for numerical features
16
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
17
fig.subplots_adjust(hspace=0.4, wspace=0.4)
18
​
19
for i, col in enumerate(numerical_cols):
20
    sns.histplot(data[col], kde=True, ax=axes[i//2, i%2])
21
    axes[i//2, i%2].set_title(f'Distribution of {col}', fontsize=10)
22
    axes[i//2, i%2].set_xlabel('')
23
    axes[i//2, i%2].set_ylabel('')
24
​
25
print(descriptive_stats)
       Number_of_Riders  Number_of_Drivers  Number_of_Past_Rides  \
count       1000.000000        1000.000000           1000.000000   
mean          60.372000          27.076000             50.031000   
std           23.701506          19.068346             29.313774   
min           20.000000           5.000000              0.000000   
25%           40.000000          11.000000             25.000000   
50%           60.000000          22.000000             51.000000   
75%           81.000000          38.000000             75.000000   
max          100.000000          89.000000            100.000000   

       Average_Ratings  Expected_Ride_Duration  Historical_Cost_of_Ride  
count      1000.000000              1000.00000              1000.000000  
mean          4.257220                99.58800               372.502623  
std           0.435781                49.16545               187.158756  
min           3.500000                10.00000                25.993449  
25%           3.870000                59.75000               221.365202  
50%           4.270000               102.00000               362.019426  
75%           4.632500               143.00000               510.497504  
max           5.000000               180.00000               836.116419 
EDA for feature selection: Univariate Analysis of numerical variables
The descriptive statistics and distributions for the numerical features provide the following insights:

Number_of_Riders: The distribution is fairly uniform, with values ranging from 20 to 100 riders. It suggests a wide variation in the number of riders per ride.
Number_of_Drivers: This feature also shows a wide range, but it is skewed towards lower numbers, indicating that there are often fewer drivers available compared to riders.
Number_of_Past_Rides: The distribution of past rides is fairly even, though slightly skewed towards lower values. It indicates variability in customer experience with the service.
Average_Ratings: The ratings are fairly normally distributed, with a mean around 4.26, indicating generally positive feedback from customers. The minimum rating is 3.5 and the maximum is 5.0.
Expected_Ride_Duration: The expected ride duration varies significantly from 10 to 180 minutes, with a mean of approximately 99.59 minutes. The distribution is fairly uniform.
Historical_Cost_of_Ride: The cost of rides varies widely, from about 26 to 836 units, with a mean of approximately 372.50 units. The distribution is right-skewed, indicating that most rides cost less than the mean, but there are some significantly more expensive rides.
Next, we will explore the categorical features to understand their distribution and how they might relate to the target variable. We will plot the counts of each category for the categorical features and note any observations:

# plotting distributions for categorical features
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, col in enumerate(categorical_cols):
    sns.countplot(data=data, x=col, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f'Distribution of {col}', fontsize=10)
    axes[i//2, i%2].set_xlabel('')
    axes[i//2, i%2].set_ylabel('')
    axes[i//2, i%2].tick_params(axis='x', rotation=45)

plt.tight_layout()
1
# plotting distributions for categorical features
2
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
3
fig.subplots_adjust(hspace=0.4, wspace=0.4)
4
​
5
for i, col in enumerate(categorical_cols):
6
    sns.countplot(data=data, x=col, ax=axes[i//2, i%2])
7
    axes[i//2, i%2].set_title(f'Distribution of {col}', fontsize=10)
8
    axes[i//2, i%2].set_xlabel('')
9
    axes[i//2, i%2].set_ylabel('')
10
    axes[i//2, i%2].tick_params(axis='x', rotation=45)
11
​
12
plt.tight_layout()
Univariate Analysis for categorical variables
Here’s the Univariate Analysis Summary for Categorical Features:

Location_Category: The distribution across different location categories shows a variation, with “Urban” likely being the most common, followed by “Suburban” and “Rural” categories. It suggests that the location category could impact the cost of rides, potentially due to differences in demand and availability.
Customer_Loyalty_Status: There are varying levels of loyalty status among customers, including categories like “Silver”, “Regular”, and “Gold”. The distribution indicates a mix of new and loyal customers, which might influence ride costs through loyalty discounts or premium pricing.
Time_of_Booking: The time of booking varies, including “Night”, “Evening”, “Afternoon”, and “Morning”. This feature could affect ride costs due to varying demand at different times of the day.
Vehicle_Type: There’s a distribution across different vehicle types, such as “Premium” and “Economy”. The type of vehicle chosen for the ride likely impacts the cost directly, with premium vehicles costing more than economy options.
Bivariate Analysis
After analyzing the distributions of numerical and categorical features, the next step is to perform Bivariate Analysis to explore the relationships between the target variable (Historical_Cost_of_Ride) and the other features. This analysis will help us identify which features have the most significant impact on ride cost, guiding our feature selection process for predictive modelling. Let’s proceed with this:


plt.tight_layout()
plt.show()
1
num_plots = len(numerical_cols[:-1])
2
n_cols = 2
3
n_rows = (num_plots + 1) // n_cols
4
​
5
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, n_rows * 4))
6
fig.subplots_adjust(hspace=0.4, wspace=0.4)
7
​
8
axes = axes.flatten()
9
​
10
# plot each numerical column against the historical cost.
11
for i, col in enumerate(numerical_cols[:-1]):
12
    sns.scatterplot(data=data, x=col, y='Historical_Cost_of_Ride', ax=axes[i])
13
    axes[i].set_title(f'{col} vs Historical_Cost_of_Ride', fontsize=10)
14
    axes[i].set_xlabel(col)
15
    axes[i].set_ylabel('Historical_Cost_of_Ride')
16
​
17
for j in range(i + 1, n_rows * n_cols):
18
    fig.delaxes(axes[j])
19
​
20
plt.tight_layout()
21
plt.show()
EDA for Feature Selection: Bivariate Analysis of numerical features
The scatter plots of numerical features against the Historical_Cost_of_Ride reveal several insights:


Number_of_Riders: There does not appear to be a clear linear relationship between the number of riders and the cost of the ride. The distribution of points is quite scattered, suggesting that while the number of riders may influence the cost, it is not a straightforward linear relationship.
Number_of_Drivers: Similar to the number of riders, the number of drivers available does not show a clear linear relationship with the ride cost. It indicates that while driver availability might affect pricing dynamics, it does not do so in a simple, direct manner.
Number_of_Past_Rides: There’s no evident linear relationship between the number of past rides a customer has had and the cost of their rides. It suggests that customer loyalty, as measured by the number of past rides, does not directly influence the cost of rides linearly.
Average_Ratings: The scatter plot does not indicate a strong linear relationship between average ratings and ride cost. While we might have expected higher-rated drivers or customers to be associated with higher costs, the data does not support this.
Expected_Ride_Duration: There seems to be a more noticeable relationship here, with a potential trend indicating that longer expected ride durations are associated with higher costs. It is intuitive, as longer rides would naturally cost more.
Given these observations, Expected_Ride_Duration stands out as a numerical feature with a more discernible relationship to the Historical_Cost_of_Ride. The other numerical features do not show clear linear relationships with the ride cost, but they might still contribute information in combination with other features or through non-linear relationships.

Next, let’s examine how the categorical features relate to the Historical_Cost_of_Ride using box plots to explore the variance in ride costs across different categories. It will help identify if certain categories are consistently associated with higher or lower ride costs:

# bivariate Analysis: categorical features vs historical_cost_of_ride
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, col in enumerate(categorical_cols):
    sns.boxplot(data=data, x=col, y='Historical_Cost_of_Ride', ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f'{col} vs Historical_Cost_of_Ride', fontsize=10)
    axes[i//2, i%2].set_xlabel('')
    axes[i//2, i%2].set_ylabel('Historical_Cost_of_Ride')
    axes[i//2, i%2].tick_params(axis='x', rotation=45)

plt.tight_layout()
1
# bivariate Analysis: categorical features vs historical_cost_of_ride
2
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
3
fig.subplots_adjust(hspace=0.4, wspace=0.4)
4
​
5
for i, col in enumerate(categorical_cols):
6
    sns.boxplot(data=data, x=col, y='Historical_Cost_of_Ride', ax=axes[i//2, i%2])
7
    axes[i//2, i%2].set_title(f'{col} vs Historical_Cost_of_Ride', fontsize=10)
8
    axes[i//2, i%2].set_xlabel('')
9
    axes[i//2, i%2].set_ylabel('Historical_Cost_of_Ride')
10
    axes[i//2, i%2].tick_params(axis='x', rotation=45)
11
​
12
plt.tight_layout()
Bivariate Analysis of categorical variables
The box plots reveal how the Historical_Cost_of_Ride varies across different categories for each categorical feature:

Location_Category: Ride costs vary significantly by location category, with “Urban” locations generally showing a wider range and potentially higher median costs compared to “Suburban” and “Rural” areas. It suggests that location plays a crucial role in determining ride costs, likely due to differences in demand, availability, and operational costs.
Customer_Loyalty_Status: There are noticeable differences in ride costs based on the loyalty status of the customer. While the median costs across different statuses seem relatively close, the variability in costs suggests that loyalty status could influence pricing, possibly through loyalty discounts or perks for higher-status customers.
Time_of_Booking: The time of booking appears to impact ride costs, with certain times of day showing different cost distributions. It could reflect demand variations throughout the day, with peak times potentially commanding higher prices.
Vehicle_Type: As expected, the type of vehicle has a significant impact on the cost of rides. “Premium” vehicles generally have higher costs compared to “Economy” vehicles, indicating that vehicle type is a critical factor in pricing.
Insights Gathered from EDA for Feature Selection
Based on the EDA, the following features are identified as potentially important for predicting ride costs:

Expected_Ride_Duration: Shows a clear relationship with ride cost.
Location_Category: Significant variance in costs by location.
Customer_Loyalty_Status: Variability in costs suggests an impact on pricing.
Time_of_Booking: Variations in costs indicate an influence of demand at different times.
Vehicle_Type: Directly influences cost with clear distinctions between vehicle types.
The other numerical features (Number_of_Riders, Number_of_Drivers, Number_of_Past_Rides, Average_Ratings) do not show strong linear relationships with the target variable but could still be useful, especially when combined with other features or through engineered features that capture non-linear relationships or interactions.

Other than these features, you are always free to select and create new features based on domain knowledge.


Summary
So, this is how you can perform EDA for Feature Selection using Python. EDA is a valuable step in a data science workflow, particularly for feature selection. It helps you know about the best candidates for features based on their relationship with the target variable and each other, their relevance, and their predictive power.
