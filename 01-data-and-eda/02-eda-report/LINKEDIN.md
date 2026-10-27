Draft only. Not published.

A random split of these bike-share hours looked more accurate than the next year. I fit the same ridge model both ways. On a mixed holdout it scored RMSE 110.56838688362778. On 2012, after training only on 2011, it scored 158.0887211276591. The random training fold contained 4344 hours from 2012. That is not a forecast.

The level shift is the reason. Mean rentals moved from 143.79444765760556 in 2011 to 234.6663613464621 in 2012. The hour-of-day shape barely changed: the correlation of the 24 hourly means was 0.9992071561496558. A check of the hour column alone would have said the timing was stable. The demand at those hours was not.

Temperature is the other trap. Its correlation with rentals is 0.4047722757786589, and a temperature-only model still lost to a calendar model on the next year. Casual rentals plus registered rentals equal the target in every row. A model allowed to see casual looked better. It was reading part of the label.

If the split contains the period you claim to be predicting, do not take the score to a stakeholder.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/01-data-and-eda/02-eda-report

#MachineLearning #DataScience #ExploratoryAnalysis #Python
