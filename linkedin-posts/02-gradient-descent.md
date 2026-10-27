Draft only. Not published.

I set the batch step to 2.5 times the largest stable step for a least-squares fit. After 22 epochs the training loss was 967634.146272341, and the next step crossed one million. The same fit with the safe step was within 1% of the least-squares loss by epoch 27.

The data are the UCI wine measurements. I used the other 12 assays to fill in alcohol. Predicting the training mean had held-out mean squared error 0.7444. Least squares got 0.4190. Batch gradient descent, momentum, and Adam at learning rate 0.05 all landed on that least-squares number. Mini-batch gradient descent, given the learning rate that was safe for the full batch, finished at 0.5068. Adam at the paper's default of 0.001 was still short after 250 passes, at 0.4442.

A lower loss is not automatically a closer answer. On a simulated valley with condition number 361.7, Adam cut the loss from 3.7755 to 0.1347 and ended farther from the least-squares weights than it started. I would not ship the mini-batch run just because it used the same optimizer family.

Tune the step to the batch you actually use. Do not crown an optimizer from one bowl.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/00-foundations/02-gradient-descent

#MachineLearning #Optimization #GradientDescent #Python
