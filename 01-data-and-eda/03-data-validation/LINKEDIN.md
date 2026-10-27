Draft only. Not published.

A null check passed every row of a census test file whose income labels were all wrong. After stripping spaces, all 16281 labels still ended in a period. The reported schema rejected all 16281. The null-only schema rejected none. There were 0 null cells.

I then scored the file anyway, with the label repaired only so the metric was defined and the features left padded. Log loss moved from 0.31833658576426743 on the cleaned file to 0.425486700706032. The paired difference was 0.10715011494176453, and the percentile interval excluded zero.

The opposite failure is quieter. I replaced 814 workclass values with a token that is not in the donor list. The schema rejected those rows. The log-loss interval included zero. A score that does not move is not evidence that the file is valid.

If the check only looks for nulls, it will wave through a bad label. If you wait for the score to move, you will miss a bad token.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/01-data-and-eda/03-data-validation

#MachineLearning #DataScience #DataValidation #Python
