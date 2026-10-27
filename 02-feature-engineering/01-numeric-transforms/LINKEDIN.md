Draft only. Not published.

The model that won my selection rule was a degree-3 polynomial. The model that taught the lesson was a logarithm of one column.

I fit ridge regressions to a public concrete-mix file and predicted compressive strength in MPa. The split kept every later break test of a training recipe out of the holdout. A training-mean baseline had holdout RMSE 15.664861218083027. Scaling the raw inputs did not improve group CV. Logging age, the only column whose training skew cleared the pre-registered threshold, did. Yeo-Johnson and a shifted Box-Cox, both applied to every column, lost to that one log. Unshifted Box-Cox could not fit. Slag, fly ash, and superplasticizer contain zeros.

The polynomial had the lowest group CV RMSE, so the script reports it. Its holdout RMSE was 6.190033630528194, on 164 columns. Against the age log, a 1000-resample bootstrap of that holdout gap had a lower end of -0.06188102255652603, so it included zero. A lower score is not the same thing as a transform you can explain to the lab.

If a column is skewed and nonnegative, try the log before a general power transform. Write the selection rule before you read the holdout.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/02-feature-engineering/01-numeric-transforms

#MachineLearning #DataScience #FeatureEngineering #Python
