# Classification-of-Machine-Failure
Creating a Model to Predict the failure of Machine based on various attributes using various Machine Learning Algorithms and deploying the model using StreamLit

## WorkFlow :-

![image](https://user-images.githubusercontent.com/87432703/139031791-499a921c-ba40-46c5-9190-a504d0bb59cb.png)

## Steps Followed: 
1.) Exploratory Data Analysis : followed the Steps of data pre-processing by doing the conversion of data types required
followed by changing the column names for easy access of the attributes

2.) Data Visualization : checked for the relationship of each attributes among each other and with the target vairables by doing
various visualizations like 
a.) Pairplot 
b.) Heatmap 
c.) Pie charts for various submodes 
d.) Box Plots for Failure rates 

3.) Feature Engineering : Checking for outliers. Plot the attributes using Boxplot and Histograms in order to get the 
outliers and the skewness of the attributes respectively

__Methods used for Handling Outliers are :__

a.)Z-score test

b.) Inter-Quantile Range

4.) Data Transformation : Standardizing the data using standard scalar

5.) Data Balancing : The data set is balanced as the target variable had an imbalanced data which was more for the machine not failing
so if build the model without balancing then the predictions will be biased more towards the Machine Not failing prospect due to high value.
so we did oversampling of our data set using SMOTE library

6.) Model Building : We build the Model using various machine learning Algorithms with and without sampled data in order to compare our model for 
best accuracy 

a.) Model building without sampled data 

![image](https://user-images.githubusercontent.com/87432703/139035898-80e8635f-531f-42f6-8d8c-8efc856da9c4.png)

b.) Model building with Sampled data 

![image](https://user-images.githubusercontent.com/87432703/139036154-caf1d826-e339-4580-ad5a-f017acc1b5c7.png)


7.) Hyper Parameters Tuning : hyper parameters gives us a generalized model so we did tuning for various models and achived 
highest accuracy For Random Forest 

![image](https://user-images.githubusercontent.com/87432703/139036438-420e5d69-7c99-41eb-8107-b9086d4f78c6.png)

8.) Deployment : Deployed the model using Streamlit which is an open source Pyhton library
did deployment for entire test data as well as single values for better and quick results

![image](https://user-images.githubusercontent.com/87432703/139036798-7738f6fe-3f71-460d-9715-3a3c6220aad9.png)

