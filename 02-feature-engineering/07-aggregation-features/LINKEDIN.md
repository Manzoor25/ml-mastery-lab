A feature that is the answer will look like a model. I aggregated a UK gift retailer's 2009-2010 invoices to one row per customer and asked which summary predicts spend in a locked 90-day window. Development RMSE selected recency, invoice count, and past spend. On held-out customers that recipe scored 1392.581556339998. Predicting the training mean scored 2326.8056756196756.

The failure was the basket. Distinct products, mean line value, and mean quantity beat the training mean on the estimation customers and lost on the development customers. Averages are not a total. Past spend already carried most of the signal. Adding recency and invoice count won development RMSE, and the holdout gap against past spend alone included zero. I did not switch after seeing that past spend had the lower holdout error.

The illegal feature is the label itself. That ridge scored 1.1496075472434353. The interval excluded zero. A sum through the end of the file also beat the honest point estimate, and its interval included zero. It was still ineligible. Those future invoices did not exist on the cutoff morning.

If the column contains the spend you are trying to predict, the error is not a forecast.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/02-feature-engineering/07-aggregation-features

#MachineLearning #DataScience #FeatureEngineering #DataLeakage #Python
