Draft only. Not published.

A fully grown tree fit the training wines with mean squared error 0. On the holdout it scored 0.6144, worse than predicting the training mean, which scored 0.6044. The linear model, chosen on validation splits that never saw those rows, scored 0.4024.

The target was red-wine quality, from 11 chemical tests. Higher-degree polynomials looked better on the training rows and then failed. A degree-4 expansion had validation mean squared error 14463.492598751998. A ridge penalty I fixed in advance, not tuned on the test set, brought the holdout score to 0.4474 and still lost to the linear model.

The simulation made the split obvious. On a sine curve with known noise, a degree-3 fit had fresh-data mean squared error 0.3384. Degree 15 cut the training error and produced a fresh-data error of 22253291766963.31. A lower training number was not a better model.

Choose complexity on data the final score has not seen. If the flexible model cannot beat the mean, do not ship it.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/00-foundations/03-bias-variance

#MachineLearning #BiasVariance #ModelSelection #Python
