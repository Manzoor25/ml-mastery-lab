Draft only. Not published.

I ran PCA on the UCI wine measurements without scaling. One component held 0.9981 of the variance, and its dominant loading was proline at 0.9998. Proline's standard deviation is 314.9075. Alcohol's is 0.8118. I had mostly measured the unit, not the chemistry.

After I z-scored each column, the first two components explained 0.5541 of the variance. That is the number I would take into a meeting. The 0.9981 figure is what you get when you forget that variance lives in squared units.

The same table is a lesson about testing every column. Class 0 has higher alcohol than class 1. The mean gap is 1.4660, and a 10,000-draw bootstrap 95% interval runs from 1.2897 to 1.6354. Hue does not separate those two cultivars. Its Welch p-value is 0.8402. I then shuffled the cultivar labels and tested all 13 features at 0.05. At least one feature looked significant in 0.387 of 1,000 simulations. Each individual test was aimed at 0.05. The chance of a false alarm somewhere in the family was not.

The interval excludes zero, so this particular gap is stable if I resample these rows. It does not mean the source file stated a unit. It does not. Scale a column before you call it the main direction. Name the comparison before you rank p-values. A small p-value on a scanned column is not a finding.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/00-foundations/01-linalg-probability-stats

#MachineLearning #Statistics #DataScience #Python
