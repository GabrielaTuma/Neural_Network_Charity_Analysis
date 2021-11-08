# Neural_Network_Charity_Analysis
Module 19 - Deep Learning 

## Project Overview

Using machine learning and neural networks, this project's goal is to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, a CSV containing more than 34,000 organizations that have received funding over the years is going to be the input. Within this dataset are a number of columns that capture metadata about each organization:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

#### Deliverable 1

- [x] 1. The EIN and NAME columns have been dropped
- [x] 2. The columns with more than 10 unique values have been grouped together
- [x] 3. The categorical variables have been encoded using one-hot encoding
- [x] 4. The preprocessed data is split into features and target arrays
- [x] 5. The preprocessed data is split into training and testing datasets
- [x] 6. The numerical values have been standardized using the StandardScaler() module


#### Deliverable 2

- [x] 1. The number of layers, the number of neurons per layer, and activation function are defined
- [x] 2. An output layer with an activation function is created
- [x] 3. There is an output for the structure of the model
- [x] 4. There is an output of the model’s loss and accuracy
- [x] 5. The model's weights are saved every 5 epochs
- [x] 6. The results are saved to an HDF5 file

[AlphabetSoupCharity.ipynb file](https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/AlphabetSoupCharity.ipynb) 

[AlphabetSoupCharity_Optimization.h5 file](https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/tree/main/AlphabetSoupCharity_Optimization.h5) 

#### Deliverable 3

The model is optimized, and the predictive accuracy is increased to over 75%, or there is working code that makes three attempts to increase model performance using the following steps:
- [x] 1. Noisy variables are removed from features
- [x] 2. Additional neurons are added to hidden layers
- [x] 3. Additional hidden layers are added
- [x] 4. The activation function of hidden layers or output layers is changed for optimization
- [x] 5. The model's weights are saved every 5 epochs
- [x] 6. The results are saved to an HDF5 file

[AlphabetSoupCharity_Optimzation.ipynb file](https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/AlphabetSoupCharity_Optimzation.ipynb) 

[AlphabetSoupCharity.h5 file](https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/tree/main/AlphabetSoupCharity.h5) 


#### Deliverable 4

- [x] 1. There is a title, and there are multiple sections
- [x] 2. Each section has a heading and subheading
- [x] 3. Links to images are working, and code is formatted and displayed correctly
- [x] 4. The purpose of this analysis is well defined
- [x] 5. There is a bulleted list that answers all six questions
- [x] 6. There is a summary of the results
- [x] 7. There is a recommendation on using a different model to solve the classification problem, and justification



## Resources

- Software: Python 3.7.6, Visual Studio Code 1.58.1, Jupyter Notebook 6.3.0


## Results

- Before using any model of prediction in our dataset a target needs to be defined. Looking at the list of columns and meanings it was possible to find the perfect target for the analysis: **IS_SUCCESSFUL—Was the money used effectively**

- All other columns can be considered features for the model except EIN and NAME, that are identification columns and should be removed during preprocessing 

We can see the list of categorical variables and their counts for unique values (that will become columns later):

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Deliverable%202%20cat%20list%20.png">
</kbd>  &nbsp;
</p>

After preprocessing a deep learning model was generated with the following settings:

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Deliverable%202%20summary.png">
</kbd>  &nbsp;
</p>

Running the model with 100 epochs an accuracy of 0.7262 was calculated:

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Deliverable%202%20accuracy.png">
</kbd>  &nbsp;
</p>

#### Deliverable 3 

Optimize your model in order to achieve a target predictive accuracy higher than 75% by using any or all of the following:

- Dropping more or fewer columns

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Simplifyed%20cat%20list.png">
</kbd>  &nbsp;
</p>


<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Simplifyed%20accuracy.png">
</kbd>  &nbsp;
</p>

- Creating more bins for rare occurrences in columns

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Detailed%20APP%2050%20.png">
</kbd>  &nbsp;
</p>

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Detailed%20CLASS%20100.png">
</kbd>  &nbsp;
</p>

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Detailed%20cat%20list.png">
</kbd>  &nbsp;
</p>

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Detailed%20accuracy%20.png">
</kbd>  &nbsp;
</p>

- Increasing or decreasing the number of values for each bin

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Simplifyed%20APP%201000.png">
</kbd>  &nbsp;
</p>

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Simplifyed%20CLASS%203000.png">
</kbd>  &nbsp;
</p>

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Simplifyed%20accuracy.png">
</kbd>  &nbsp;
</p>


- Adding more neurons to a hidden layer

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Deep%20learning%20units=200.png">
</kbd>  &nbsp;
</p>

- Adding more hidden layers

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Deep%20learning%203%20hidden%20layers%20.png">
</kbd>  &nbsp;
</p>


- Adding or reducing the number of epochs to the training regimen

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Deep%20learning%20200%20epochs%20.png">
</kbd>  &nbsp;
</p>

- Other models - Logistic Regression

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Logistic%20Regression.png">
</kbd>  &nbsp;
</p>

- Other models - Random Forest

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Random%20Forest.png">
</kbd>  &nbsp;
</p>


Many manipulations were made in order to comprehend how the different settings could affect the model's accuracy. The best result was found using the following settings:
- batch_size = 32 
- epochs = 30
- 3 hidden layers using relu, output using sigmoid

Model summary:

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Deliverable%203%20model%20.png">
</kbd>  &nbsp;
</p>

Accuracy was improved, but not significantly, model was not able to predict with 75% or more accuracy.

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Deliverable%203%20accuracy.png">
</kbd>  &nbsp;
</p>



## Summary

While adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model many results were generated. The first approach was to simplify the columns, filter the active status organizations (big majority) and categorize the income amount column into with or without income. 
 
<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Simplifyed%20accuracy.png">
</kbd>  &nbsp;
</p>

The result was a very low accuracy, that surprisingly was not caused by the manipulation done on the income column, but actually on the status columns. 

<kbd>
  <img src="https://github.com/GabrielaTuma/Neural_Network_Charity_Analysis/blob/2f97af0c9f8c9eb18d119e55af8e582d59d75a27/Resources/Images%20/Status%20count.png">
</kbd>  &nbsp;
</p>

Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.

75% can be considered an educated guess, but more needs to be done to create a prediction model. Looking 

