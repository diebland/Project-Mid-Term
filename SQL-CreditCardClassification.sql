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
SELECT AVG(average_balance) FROM credit_card_data;

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



