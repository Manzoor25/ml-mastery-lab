Draft only. Not published.

I opened the published training file and the published test file and found no income label in common. All 16281 test labels still had a trailing period after I stripped the spaces. An exact string match was impossible before any model ran.

The expensive bug was quieter. I stripped spaces on the training categories and left them on the test file. All 97686 categorical test cells became unknown to the encoder. Log loss on the published test file moved from 0.3207506692781671 to 0.42049919201950064. The numeric columns were still aligned. The categories were not.

A cleanup that looked responsible was worse. The interquartile fence on capital gain collapsed to 0.0 because most rows have no gain. Dropping the rows outside that fence removed every positive gain from training, and log loss rose to 0.34962426376625366.

Missing occupation was informative, not a hole to paper over. The positive rate was 0.10363537710255019 when occupation was missing and 0.24903965101894654 when it was present. Filling the token with the mode throws that difference away.

If a cleanup rule deletes the rare rows, check what those rows were before you trust the smaller file.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/01-data-and-eda/01-messy-data-cleanup

#MachineLearning #DataCleaning #MissingData #Python
