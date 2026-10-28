One sine and cosine pair is not a model of a commute. Westbound I-94 volume rises in the morning and peaks again in the afternoon. A single daily wave cannot draw both.

I held out the later hours before fitting anything. The reported ridge forecast, chosen by training-period time-series CV, had RMSE 493.8430276959893. Predicting the training mean on every hour scored 1970.206195090623. Yesterday at the same hour scored 1038.3391488036405. Two harmonics beat one harmonic. One-hot hour beat both of those calendar fits. Lags did more than either calendar encoding. I report the pair with the lowest training CV error, not the lowest holdout error.

A rolling mean that includes the current hour scored 492.7845532696366. That is not a forecast. The hour's own volume is the target. A date-cluster bootstrap of the gap included zero, so the leak did not earn a win. A row shift that treats the previous stored row as one hour ago was worse, including across a hole of 7387.0 hours.

If the sensor skipped an hour, do not call the last row lag 1.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/02-feature-engineering/04-datetime-features

#MachineLearning #DataScience #FeatureEngineering #TimeSeries #Python
