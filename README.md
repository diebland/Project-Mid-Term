# Classification Problem : Credit card offer analysis


## Background

Apart from the other banking and loan services, the bank provides credit card services which is a very important source of revenue.

The bank wants to understand the demographics and other characteristics of the customers accepting a credit card offer and of those not accepting the credit card offer.

Then, the bank designs a focused marketing study, with 18,000 current bank customers including both customers with and without credit card.  This focused approach allows the bank to analyze data for who does and does not respond to the offer, and to use existing demographic data that is already available on each customer.


Data: The data set consists of information on 18,000 current bank customers. These are the definitions of data points provided:

- **Customer Number**: A sequential number assigned to the customers (this column is hidden and excluded – this unique identifier will not be used directly).
- **Offer Accepted**: Did the customer accept (Yes) or reject (No) the offer. 
- **Reward:** The type of reward program offered for the card.
- **Mailer Type**: Letter or postcard.
- **Income Level**: Low, Medium or High.
- **#Bank Accounts Open**: How many non-credit-card accounts are held by the customer.
- **Overdraft Protection**: Does the customer have overdraft protection on their checking account(s) (Yes or No).
- **Credit Rating**: Low, Medium or High.
- **#Credit Cards Held**: The number of credit cards held at the bank.
- **#Homes Owned**: The number of homes owned by the customer.
- **Household Size**: Number of individuals in the family.
- **Own Your Home**: Does the customer own their home? (Yes or No).
- **Average Balance**: Average account balance (across all accounts over time). 
- **Q1, Q2, Q3 and Q4** **Balance**: Average balance for each quarter in the last year


## Objectives

The task is to build a model that will provide insight into why some bank customers accept credit card offers. There are also other potential areas of opportunities that the bank wants to understand from the data.

we will use different tools to analyze the data :

SQL, tableau and statistics analysis with Python.

## Our steps

1. SQL 

The project includes SQL part to explore the dataset we have been provided with.

We first need to create our database and import the data on MySQLWorkbench.
We will write several queries using GROUP BY and ORDER BY statements windows functions, subqueries and views.
Link to the file.


2. Tableau

We have used Tableau to better vizualize the data.
The dashbord contains different bar charts and tables to see correlation or absence of correlation between some of our variables, especially our target column.

link of the file or include images

3. Classification model

3.1. Exploring the data
3.2 Modeling

We will use different models to compare the accuracies and find the model that best fits our data. 
Then, we will compare the different models using accuracy measures.

4. Bonus : Write three or more functions that can be reused for another project. This work can be found on the file : link of the file

## Conclusion


## Documentation

Original and cleaned datasets as well as the graphs used in this project can be found on the folder documentation

Sources https://www.kaggle.com/ekrembayar/fifa-21-complete-player-dataset)

List of libraries (with a link to the documentation) https://scikit-learn.org/stable/index.html

https://docs.scipy.org/doc/scipy/reference/index.html

https://pandas.pydata.org/

https://numpy.org/doc/

https://matplotlib.org/3.1.1/

https://seaborn.pydata.org


