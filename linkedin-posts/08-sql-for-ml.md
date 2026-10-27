Draft only. Not published.

A repeat-purchase query that was allowed to see the outcome scored a test log loss of 0.03506553353837828. The same logistic model, restricted to invoices before the cutoff, scored 0.6063879510172124. I am not shipping the first number.

I built the label from a public gift-retailer invoice file. A customer was positive after a purchase on or after the cutoff. Honest features used only earlier purchase lines: orders, spend, products, cancellations, and recency. Extending the date filter through the end of the extract changed the order count for every positive customer, all 1952 of them. The holdout loss fell because those features had started to contain the label.

The quieter failure is the split. A random split of history invoices put 1254 of 3317 eligible customers on both sides. A customer is not an invoice. If the split unit is smaller than the person you will score, the holdout is not a holdout.

A lower score is not a reason to drop a date filter. If the feature uses rows from the window you are trying to predict, you do not have a forecast.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/01-data-and-eda/04-sql-for-ml

#MachineLearning #DataScience #SQL #FeatureEngineering
