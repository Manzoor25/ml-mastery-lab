One-hot encoding the diagnosis codes looked like an upgrade. It built 2386 columns. Under a patient-grouped cross-validation, that logistic regression had a worse log loss than the same model with those codes left out.

The encoding that won the rule I wrote down first was a smoothed target mean, with the patient's other encounters held out of the mean. Holdout log loss was 0.32616331460860226. Predicting the training early-readmission rate on every row scored 0.33981780404157347. A patient bootstrap of that gap stayed positive.

I fit the raw means on the same rows I then scored. That run posted a cross-validated ROC-AUC of 0.6846182227873063. It was marked ineligible before I looked at the table. On the holdout it lost to the model I kept. Accuracy was the other trap. At a threshold of one half, the selected model was less accurate than predicting that nobody would be readmitted early. Almost every probability sat below that line. Log loss moved. Accuracy did not.

A discharge code with 1642 rows and an early-readmission rate of 0.0 never entered the fit. A zero-rate code is a shortcut, not a diagnosis effect.

If a column has hundreds of levels, do not one-hot it by default, and do not let a category mean see the label of the row you are about to score.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/02-feature-engineering/02-categorical-encoding

#MachineLearning #DataScience #FeatureEngineering #Python
