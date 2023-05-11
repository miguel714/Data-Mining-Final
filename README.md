# Data-Mining-Final-Project

# Final Project 

# 1. Introduction

# 1.1 Task Description: What are you doing?

# For my Final Project, I will be creating four different files of coding that are most suited for the excel sheet that was provided. The four coding files are PCA (Principal Component Analysis), Multiple Linear Regression, K-Nearest Neighbors, and Neural Networks. These four coding files will help me understand the similarities and differences between the customers and the relationships between the customer’s data. The end goal is to provide guidance to the business owner on how to optimize their sales strategies and improve customer satisfaction. 

# 1.2 Data Description: The grain of the data, scale, units: 

# The grain of the data are the four variables which are NoOfSalesCalls, NoOfTargetedEmails, NoOfSales, and CustomerSatisfaction. The scale of the dataset varies for each column. For example, the Customer column ranges from 1 to 47, NoOfSalesCalls column ranges from 0 to 5, NoOfTargetedEmails column ranges from 0 to 3,  NoOfSales column ranges from 0 to 4, CustomerSatisfaction column ranges from -1 to 1. The units of the dataset are numerical values and the assumption for CustomerSatisfaction are ratings, and so they are represented by numbers.

# 2. Data Preparation

# 2.1 Data Exploration: 

# <img width="572" alt="Screenshot 2023-05-11 at 1 03 16 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/56c66f8a-cd57-4369-8221-7875ca615339">

# The image above is to import and then read the csv file into a jupyter notebook. I then printed the csv in order to see what the csv file contains. 

# <img width="525" alt="Screenshot 2023-05-11 at 1 04 04 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/9188e13c-15cf-4c40-b5f4-53daf0339c39">

# The next image above represents the shape of the data in order to know how I will fit and train the data when I start to apply certain algorithms for the dataset. 

# <img width="622" alt="Screenshot 2023-05-11 at 1 05 13 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/1f441a0b-95bb-433b-a6fb-484de3933d3e">

# The last image above represents the first 5 rows of the dataset to get a glimpse of the dataset. 

# These techniques for data exploration are crucial in order to get an idea of how the dataset is set up. This is a preparation before we start preprocessing, fit and training, and then modeling the dataset, so by exploring the dataset that was provided in order to get a better understanding of its structure and the distribution of the variables within the dataset. What I found in the dataset is that there are no missing values. Also, I found that there was a strong correlation between the number of sales calls that were made to the customer and the number of sales made to that customer. Below are examples showing how I explored the dataset. 

# 2.2 Data Visualization: 

# The dataset that was provided is numerical data, therefore we can use certain algorithms that deal with numerical values, such as KNN, Multiple Linear Regression, Neural Networks, and PCA . As part of the visualization, it has critical components of the data analysis which provides a course of action to visually explore and interpret the patterns and relationships within the data. For example, we can visualize the data on the number of sales calls and emails sent which can be used to assess effectiveness of the sales strategy by looking at the number of sales made. There are several types of visualizations that can be used depending on the nature of the data the question being asked. So by using a combination of the visualizations, I can gain a better understanding of the data and to gain insight that is shown in the data. 

# 2.3 Preprocessing:

# <img width="621" alt="Screenshot 2023-05-11 at 1 06 38 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/f2260c85-7704-49f6-a4c3-0a5b2e16421b">

# For the first coding file, which is about PCA, I normalized four variables which were NoOfSalesCalls, NoOfTargetedEmails, NoOfSales, and CustomerSatisfaction in order for it to be consistent and make them compatible with each other. Then I preprocessed it by fitting and scaling the data as a preparation before we print the summary of the PCs. 

# <img width="620" alt="Screenshot 2023-05-11 at 1 07 23 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/ac8ada5a-e0cb-49a9-870e-524ad9c50c68">

# This is the second coding file, which is about Multiple Linear Regression, so for this part I created X and y where X has the predictors and the y is the target. Then I train and split the data before running the algorithm. 

# <img width="623" alt="Screenshot 2023-05-11 at 1 07 52 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/1df636bb-7540-4879-98e4-0ab775604741">

# <img width="619" alt="Screenshot 2023-05-11 at 1 08 13 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/6d66a0fc-980c-43b8-8940-0429e84b0157">

# The third coding file, which is about KNN, is where I separated the predictors, using the loc function and the target being the ‘y’. Then, I used standardscaler to scale each variable to unit variance and after to fit and transform the data. Once that was done, I used the iloc function in order to shape the values before performing the fit and train function, and printed the new shape in order to get an idea of what it looks like after.
 
# <img width="624" alt="Screenshot 2023-05-11 at 1 08 50 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/15ea1579-8fa8-4855-a9e9-2e0da34679b7">

# The last coding file, which is about Neural Network, where I separated the predictors and the target and then I added the MinMaxScaler function in order to scale each function to a given range. 

# 3. Feature generation and transformation: scaling, dimension reduction etc. 

# Creating new features would include putting all four column variables, NoOfSalesCalls, NoOfTargetedEmails, NoOfSales, and CustomerSatisfaction, together into one predictor in order to scale, transform, and fit and train the predictor along with the target. Specifically, two of the features are the MinMaxScaler, StandardScaler, and Normalizer that I used for my coding. This is important because we need to capture the relationship between the predictor and target variable. This will help me analyze the results after we run the model. I also used the dimension reduction for the first coding file, PCA, in order to reduce the set of numerical variables. Since the dataset contained numerical values that were repetitive, we needed to remove the overlap of information. This would allow us to get the most important information from the smallest number of numerical variables. These features are necessary since they improve the quality of the results and the insights I will gain as well. 

# 4. Model development

# One of the models I developed was a Principal Component Analysis (PCA) which was performed on the four variables from the Sales of Medical Devices dataset. The four variables were NoOfSalesCalls, NoOfTargetedEmails, NoOfSales, and CustomerSatisfaction. I then normalized the four variables to be between 0 and 1 using the min-max scaling. Afterwards, the PCA that is performing will use the PCA function in order to return the four components. Now that the data is standardized and normalized, it will be passed on to the PCA function in order to transform it into principal components. Lastly, I created a summary of the dataframe in order to display the standard deviation, proportion of the variance, and the cumulative proportion of the variance for each four principal components. 

# The next model I developed involved a multiple linear regression which set the four variables,  NoOfSalesCalls, NoOfTargetedEmails, NoOfSales, and CustomerSatisfaction, as ‘X’ and Customer as ‘Y’. I then split the data into training and test sets by using the ‘train_test_split’ function. The test size was 0.2 which means 20% of the data and set the random state to 42 for reproducibility. Afterwards, I added a constant column to the training and test set in order for the linear regression models to have an intercept. I then fit an OLS (Ordinary Least Squares) linear regression model to the training set in order to predict the customer variable based on the data of the other variables. I fitted another OLS linear regression model, but with a different syntax in order to specify the formula. I printed the model’s coefficients for the both OLS models in order to get the results. Lastly, I fitted two regularization models (Ridge and LASSO) to the data and printed the regression results for both models in order to get the summary of the results, which includes the R-Squared Values RMSE (Root Mean Squared Error) and the MSE (Mean Squared Error). 

# The next model to be developed involves the use of the K-Nearest Neighbors (KNN) algorithm. So I prepared the data by splitting into predictors and target variables by using the iloc function. The predictors, NoOfSalesCalls, NoOfTargetedEmails, NoOfSales, and CustomerSatisfaction, were then scaled by using the StandardScaler in order to make sure all the features are on the same scale. Then I used a train_test_split function in order to split into training and validating sets with a test size of 0.2 and a random state of 42. The KNN model was then trained on the training set for k values ranging from 1 to 5. For each value of K, the MSE (Mean Squared Error) was calculated on the validation set by using the mean_squared_error. The MSE results for the validation set were printed for every value of K. To add another evaluation, the MSE was then calculated on the training set for every value of K. 

# The last model I developed involved using a neural network for the dataset. Similar to the other two models where the predictors, NoOfSalesCalls, NoOfTargetedEmails, NoOfSales, and CustomerSatisfaction, were given the ‘X’ variable and the ‘y’ variable being the customers. The data is then preprocessed by scaling the input features using the MinMaxScaler in order to scale the input features to range between 0 and 1. After the data is preprocessed, I continued by splitting the data using the test_train_split in order to get training and test set. The training set will be used to fit the model and the test set to evaluate the model’s performance. To initialize the neural network model, I used the MLPRegressor with a single hidden layer of two neurons, the relu activation function, the adam solver, and the iteration set to 5000. I then fitted the model to the training set and the predicted values were taken for the test set. Finally, I got the RMSE results in order to measure the difference between predicted and actual sales value. 

# 5. Results and Conclusion

# <img width="624" alt="Screenshot 2023-05-11 at 1 11 58 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/afce7990-9aa9-424d-94d5-eebb18c2820a">

# The first results from the PCA coding file shows that the first PCs explains 49.6% of the total variance of the data. The second PC explains 31.3% of the total variance, the third PC explains 10.2% of the total variance, and the last PC explains only 8.9% of the total variance. PC1 and PC2 combined explain for 80.9%, then if we add PC3 it will explain 91.1%, and PC4 will explain 100% of the variance for the data. PC1 is the most important component since it captures the majority of the variability in the data, and PC2 and PC3 explain an additional amount of variance. PC4 is the least significant as it explains the least amount of variance, therefore it has no importance. So the results can be explained by saying that the four variables,  NoOfSalesCalls, NoOfTargetedEmails, NoOfSales, and CustomerSatisfaction, are all related variables that can be summarized by the small number of principal components. 

# <img width="437" alt="Screenshot 2023-05-11 at 1 12 27 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/32cb98d0-2d9c-4e3f-870d-47f0bf7b1ccf">

# The graph above shows a scatter plot of the data that was transformed on the first two principal components. The x-axis represents PC1, which accounts for 49.5% of the variance, and the y-axis represents PC2, which accounts for 31.3% of the variance. So from what we can see is that there is some grouping along the x-axis with the values increasing from left to right. There could be a common factor driving the variance along PC1, so this could be related to the number of sales calls or sales made, since the variables had high loading values for PC1 in the PCA summary table. There are also some grouping data along the y-axis as well with the values increasing from bottom to top. Another common factor could be present as well, that is driving variation along the PC2, so it can be related to CustomerSatisfaction as the variable had the highest loading value for PC2 in the PCA summary table. Overall, there may be two main factors that are driving the variance in the data, which related to the sales calls/made and the CustomerSatisfaction. There is a lot of variation that isn’t explained by the two factors since PC1 and PC2 only account for 80% of the total variance in the data. 

# <img width="525" alt="Screenshot 2023-05-11 at 1 13 04 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/4717dd73-b065-4067-8986-a15684f0a8e0">

# After running the OLS model, these are the results that were outputted. The r-squared value of 0.016 shows that only 1.6% of the variance in the dependent variable is explained by the independent variable in the model. The adjusted R-squared has a value of 0.010 which is slightly lower, therefore it shows the model didn’t capture the relationship between the predictors and dependent variables. The F-statistic had a result of 2.463 and the p-value had a result of 0.0442. This shows that at least one of the predictor variables is statistically significant in predicting the dependent variable. However, the p-values for the individual coefficients shows that only the NoOfSales is statistically significant at the 5% significant level. The coefficients for the predictor variables show that the effect of the unit increases in each of the independent variables on the dependent variable. An example would be that the unit increase in NoOfSales is associated with 1.0572 unit increase in Customer.

# <img width="504" alt="Screenshot 2023-05-11 at 1 13 31 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/14d83f4f-4d20-4b73-9008-87f4a0a1d235">

# To explain the results for each variable, I will start with the intercept which is 22.015097 meaning that when all the independent variables are 0, then the expected value of the dependent variable is 22.015097. For the coefficient of NoOfSalesCalls it’s 0.979080, so this means that for each unit increase in the number of sales calls made, we can expect an increase of 0.979080 units in the value of the dependent variable. For the coefficient of NoOfTargetedEmails it’s -0.577289, so this means that for each unit increase in the number of targeted emails that were sent, we can expect to see a decrease of 0.577289 units in the value of the dependent variable. The coefficient of NoOfSales is 1.057182, so this means that for each unit increase in the number of sales made, we can expect an increase of 1.057182 units in the value of the dependent variable. The last coefficient of CustomerSatisfaction is -0.143139, so this means that for each unit increase in the CustomerSatisfaction rating, we can expect a decrease of 0.143139 in the value of the dependent variable. Overall, the p-values for NoOfSalesCalls, NoOfSales, and the intercept are all below 0.5, so it means that they are statistically significant. The p-values for NoOfTargetedEmails and CustomerSatisfaction are above 0.5, so this means they are not statistically significant in predicting the dependent variable which is Customer.

# <img width="520" alt="Screenshot 2023-05-11 at 1 18 39 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/f2173ef6-98a0-4755-ad22-8bab608e17cc">

# For the Ridge Regression model, the RMSE was 13.4557, meaning that the average difference between the predicted and the actual values is 13.4557. The MAE is 11.5890, so this shows that the model’s prediction had an average deviation of 11.5890 from the values. The MPE is -125.4728, so this means that the model’s predictions are on 124.4728% off from the actual value. The MAPE is 155.3383, so this means that the model’s predictions has an average percentage deviation of 155.3382. The LASSO Regression model can be explained the same way, but with different values, as shown in the output, in replace of the values of the Ridge Regression Model. So based on the results from both regression models, we can conclude that they had similar performance, but no significant difference in the prediction accuracy. However, the MAPE values could explain that both models had high percentage errors, so it means that the two models are not the best fit for this dataset. 

#<img width="293" alt="Screenshot 2023-05-11 at 1 34 42 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/0c13a9b1-a7be-4c14-ac4d-c6ebf47ea545">

# Looking at the results in the image, we see that MSE decreases as K increases from 1 to 3. It then starts to increase again for K=3 and K=5. So this is saying that the best K is 3 as it's more optimal compared to the other ones. This is because it has the lowest MSE on the validation set, so this means that it has a better performance since they show a smaller difference between predicted values and the actual values. Since K=3 has the lowest MSE of 0.34, it suggests that the model is able to predict the target variable, which is the Customers, by using the predictors, which is Sales of Medical Devices.

# <img width="620" alt="Screenshot 2023-05-11 at 1 37 05 PM" src="https://github.com/miguel714/Data-Mining-Final/assets/121070262/bf269c7e-4f1b-4ff4-b2d2-7e0a0bf17161">

# The RMSE value of 14.214 is for the predicted sales using the neural network model, which indicates the average difference between the actual sales and the predicted sales. Therefore, the RMSE indicates that the neural network’s predictions are off by approximately 14.214 units of sales. A lower RMSE value indicates better performance, but in this case, the neural network is not performing optimally, so more tuning is required to improve accuracy. 

# In Conclusion, the results from the four different algorithms that were used in the files gave mixed results. The Principal Component Analysis (PCA) and K-Nearest Neighbor seemed to give the best results based on the dataset. The objective was to guide the business owner to gain optimal performance for their sales strategies. Therefore, based on the results from the PCA, it showed that PC1 had a driving factor for why the variance is high. This could be because of SaleCalls or SalesMade as the variables had high loading variance for PC1 as seen in the summary table. Another driving factor for PC2 could be because of CustomerSatisfaction as it had a high loading variance for PC2. By looking at the graph as well, it explicitly shows the two driving factors, SaleCalls/Made and CustomerSatisfaction are the reason for driving the variance in the data, so it had a large impact. The KNN was also helpful in showing the best performance for when we runned the model. As K=3 had the lowest MSE, then it shows better performance between the predicted and actual values. By doing a test_train_split for the predictors and the target, we can find out if the performance is optimal and whether the model is able to predict the target variable, which it did. These two algorithms were best suited in order to explain how to optimize the sales strategies and improve customer satisfaction. The other two algorithms, which were Multiple Linear Regression and Neural Network, weren’t best suited for the dataset as. Lasso and Ridge Regression Model showed that MAPE had high percentage error and had similar performance, so it wasn’t the best way to help strategize sales and improve customer satisfaction. The same could be said for the Neural Network Algorithm as it had a higher RMSE since it was 14.214 off on average, and the algorithm was not performing at its best, so more tuning would need to be done. The business owner is better off finding the best results using PCA and KNN as they are best suited for the dataset and will help them optimize their sales and customer satisfaction. 














