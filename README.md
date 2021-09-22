![image](https:/Project_Details/image.png"creditCard")





# Classification Problem : Credit Card Offer Analysis


**Student Project to apply SQL, Tableau and data analysis (with Python) concepts and techniques**    




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
- **Bank Accounts Open**: How many non-credit-card accounts are held by the customer.
- **Overdraft Protection**: Does the customer have overdraft protection on their checking account(s) (Yes or No).
- **Credit Rating**: Low, Medium or High.
- **Credit Cards Held**: The number of credit cards held at the bank.
- **Homes Owned**: The number of homes owned by the customer.
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

The project includes [SQL](https://github.com/diebland/Project-Mid-Term/blob/main/SQL-CreditCardClassification.sql/ "SQL").  

We first need to create our database and import the data on MySQLWorkbench.

Then we will write several queries using GROUP BY and ORDER BY statements windows functions, subqueries and views.

    

2. Tableau

We have used [Tableau](https://github.com/diebland/Project-Mid-Term/blob/main/Tableau.twb/ "Tableau"). to explore the data and visualize the information we can extract from it.  

The dashbord contains different bar charts and tables to see correlation or absence of correlation between some of our variables, especially our target column "offer_accepted".

 

3. Classification model

[We have analyse the data and implement machine learning algorithms with Python.](https://github.com/diebland/Project-Mid-Term/blob/main/Classification_Project.ipynb/ "ClassificationModel").  


We have analyse the data and implement machine learning algorithms with Python.

3.1. Exploring the data we will  clean the dataframe and create vizualization of the datas.

3.2 Modeling

We will use two different models - Logistic Regression and KNN Classifier to compare the accuracies and find the model that best fits our data. 
Then, we will compare the different models using accuracy measures.

4. Functions 
[Please find also a Jupyter Notebook with all the functions written for this project but to be re-used for other projects.](https://github.com/diebland/Project-Mid-Term/blob/main/functions.ipynb/ "functions").  



## Conclusion

**Comparison of accuracy scores**     



| Models                     |     Logistic Regression   |   KNNClassifier | 
| --------------------------:|: -----------------------: | ---------------:|
| BoxCox & SMOTE             |           0,94            |       0,86      |
| BoxCox & upsampling        |           1               |       0,90      |
| Normalization & upsampling |           0,70            |       0,97      |     

   

We have obtained high accuracy scores in most of our runs. However, we should not necessary be confident on how our models work on future data. In fact, the high results may let us think that they could be overfitted.



## Documentation

Original and cleaned datasets as well as the graphs used in this project can be found on the folder documentation.

List of libraries (with a link to the documentation):
- [Pandas](http://https://pandas.pydata.org/"Title")  

- [Numpy](http://https://numpy.org/doc/"Title")   

- [Matplotlib](http://https://matplotlib.org/3.1.1/"Title")  

- [Seaborn](http://https://seaborn.pydata.org/"Title")   

- [Scikit-learn](http://scikit-learn.org/stable/index.html/"Title")  

- [Scipy](http://docs.scipy.org/doc/scipy/reference/index.html/"Title")


