Character n-grams beat a word vocabulary on short Reddit comments, and a vectorizer that had already seen the test comments did not get a better score.

I used the agreement-filtered GoEmotions splits and the official Ekman map. Neutral stayed neutral. A comment was kept only when every fine label mapped to one coarse emotion. That rule was locked before the fit. Development log loss then chose character n-grams over bag-of-words counts, word TF-IDF, word bigrams, and a 100-component LSA embedding. On the official test split, which did not choose the recipe, the character model had log loss 0.9546742014250085. Predicting the training class rates scored 1.5041699927045604. Word bigrams did not beat unigrams. The bootstrap interval included zero.

Fitting the unigram TF-IDF statistics on the train, development, and test text scored 0.998050295566316. That is not an improvement, and it is not eligible. The test comments had already entered the vocabulary and the inverse document frequency. A leak that fails to inflate the score is still a leak.

If the rows you are about to score helped build the vocabulary, the number is not a holdout.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/02-feature-engineering/05-text-features

#MachineLearning #DataScience #FeatureEngineering #NLP #Python
