-- Create the database and import the data
USE credit_card_classification;
-- Select all the data from table credit_card_data to check if the data was imported correctly.
SELECT * FROM credit_card_data;


-- Use the alter table command to drop the column q4_balance from the database.
-- Select all the data from the table to verify if the command worked. Limit your returned results to 10.
ALTER TABLE credit_card_data DROP q4_balance;
SELECT * FROM credit_card_data
LIMIT 10;


-- Find how many rows of data you have.
SELECT COUNT(*) FROM credit_card_data;

-- Find the unique values in 5 categorical columns:
-- `Offer_accepted`, `Reward`, `mailer_type`, `credit_cards_held`, `household_size`
SELECT COUNT(DISTINCT offer_accepted),
COUNT(DISTINCT reward),
COUNT(DISTINCT mailer_type),
COUNT(DISTINCT credit_cards_held),
COUNT(DISTINCT household_size)
FROM credit_card_data;


-- Find the average_balance of the house by decreasing order with the top 10 customers 
SELECT customer_number FROM credit_card_data
ORDER BY average_balance DESC;

-- average_balance of all the customers
SELECT ROUND(AVG(average_balance),2) FROM credit_card_data;

-- GROUP BY queries
-- average_balance of the customers grouped by `Income Level` Use an alias to change the name 'average_balance'.
SELECT income_level, AVG(average_balance) as average_balance_account FROM credit_card_data
GROUP BY income_level;
-- average_balance of the customers grouped by `number_of_bank_accounts_open`. Use an alias to change the name 'average_balance'.
SELECT bank_accounts_open, AVG(average_balance) as average_balance_account FROM credit_card_data
GROUP BY bank_accounts_open;
-- average number of credit cards held by customers for each of the credit card ratings. Use an alias to change the name 'number of credit cards held'.
SELECT credit_rating, AVG(credit_cards_held) as average_card FROM credit_card_data
GROUP BY credit_rating;
-- Is there any correlation between the columns `credit_cards_held` and `number_of_bank_accounts_open`?
-- by grouping the data by one of the variables and then aggregating the results of the other column. 
-- Visually check if there is a positive correlation or negative correlation or no correlation
SELECT bank_accounts_open, credit_cards_held, COUNT(bank_accounts_open) correlation
FROM credit_card_data
GROUP BY credit_cards_held, bank_accounts_open
ORDER BY credit_cards_held, bank_accounts_open;
-- Helding more bank accounts does not imply customers helding more credit cards. 
-- In fact, the bigger category is "customers with 1 bank but 2 credit cards".

-- You might also have to check the number of customers in each category (ie number of credit cards held) 
-- to assess if that category is well represented in the dataset to include it in your analysis. 
-- For eg. If the category is under-represented as compared to other categories, ignore that category in this analysis
SELECT credit_cards_held, COUNT(customer_number)
FROM credit_card_data
GROUP BY credit_cards_held
ORDER BY credit_cards_held;
-- Only 515 customers hold 4 credit cards. This category can be ignored for our analysis.


-- Find the customers with the following categories :
-- Credit rating medium or high
-- Credit cards held 2 or less
-- Owns their own home
-- Household size 3 or more
SELECT * FROM credit_card_data
WHERE credit_rating IN ("Medium","HIGH")
AND credit_cards_held <=2
AND own_your_home = "Yes"
AND household_size >=3;
-- filter those customers who have accepted the offer
SELECT customer_number, offer_accepted FROM credit_card_data
WHERE credit_rating IN ("Medium","HIGH")
AND credit_cards_held <=2
AND own_your_home = "Yes"
AND household_size >=3
AND offer_accepted = "Yes";


-- Find the list of customers whose average balance is less than the average balance of all the customers in the database.
SELECT customer_number
FROM credit_card_data
WHERE average_balance < (SELECT ROUND(AVG(average_balance),2) FROM credit_card_data);

-- And create a view to save this query
create or replace view Customers__Balance_View1 as
SELECT customer_number
FROM credit_card_data
WHERE average_balance < (SELECT ROUND(AVG(average_balance),2) FROM credit_card_data);

-- Find the number of people who accepted the offer vs number of people who did not.
SELECT COUNT(offer_accepted) offer_accepted FROM credit_card_data
WHERE offer_accepted = "Yes";
SELECT COUNT(offer_accepted) offer_rejected FROM credit_card_data
WHERE offer_accepted = "No";

-- Find the difference in average balances of the customers with high credit card rating and low credit card rating.

SELECT credit_rating, ROUND(AVG(average_balance),2)
FROM credit_card_data
WHERE (credit_rating = 'High' OR credit_rating = 'Medium')
GROUP BY credit_rating;


-- Find all types of communication (mailer_type) that were used and with how many customers
SELECT mailer_type, COUNT(customer_number)
FROM credit_card_data
GROUP BY mailer_type;

-- Provide the details of the customer that is the 11th least Q1_balance in your database.
SELECT * FROM (SELECT*,
DENSE_RANK() over (ORDER by q1_balance) AS Customer_with_11thleast_Q1Balance
FROM credit_card_data) subquery
WHERE Customer_with_11thleast_Q1Balance = 11;
