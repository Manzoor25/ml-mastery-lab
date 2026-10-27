Draft only. Not published.

Removing the penalty made a breast-cancer logistic model look perfect. Training log loss fell to 7.303091096676565e-11, and every training row was classified correctly. On the holdout, that same fit scored 13.33741462393542. scikit-learn's unregularized solver, given the same split, scored 92.46130322338546 and disagreed with mine by a probability of 1.0. The penalized model, with C at 1.0, is the one the two implementations agreed on. Its holdout log loss was 0.07716713894283675.

The labels were separable. Once a direction classifies every training row, the unregularized likelihood can keep increasing the coefficients. Newton and lbfgs stop at different huge vectors. A penalty is what makes the solution a thing you can check, not a race toward infinity.

The same holdout showed a quieter version of the mistake. A linear regression fit to the 0/1 labels put 36 of 114 scores outside 0 and 1. Accuracy at one half matched the logistic model. Log loss did not.

If training error vanishes and two libraries disagree, do not ship the coefficient. Match the objective, then look at a score you did not use to fit.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/00-foundations/04-linear-logistic-from-scratch

#MachineLearning #LogisticRegression #ModelEvaluation #Python
