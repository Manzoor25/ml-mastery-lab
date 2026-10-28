A duration column is the leak people name first in this bank telemarketing file. It is known only after the call. On the held-out calls I fixed before fitting, adding it did not improve the probability score.

The honest logistic regression had log loss 0.3736699669516322. Predicting the earlier subscription rate on every row scored 0.3946184390200872. The duration model scored 0.3804182462375427. Ranking did jump. A date-cluster bootstrap of the log-loss gap included zero, so I will not claim duration made the probabilities worse either. I still would not ship it. The person choosing whom to dial does not know how long the call will last.

The leak that moved log loss was training on other calls from the same campaign window. Those outcomes are not available when the next call is queued. Training on later months did the opposite. The subscription rate had risen, and the score got worse. A shuffled split was not a fair comparison. Later months are a different campaign, and that split's own ranking was near chance.

Accuracy at one half was the other trap. The honest model matched the rule that predicts nobody subscribes.

If a column is known only after the decision, leave it out, even when a ranking chart looks better. Do not read a same-period fit as proof the model learned the customer.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/02-feature-engineering/03-data-leakage

#MachineLearning #DataScience #FeatureEngineering #Python
