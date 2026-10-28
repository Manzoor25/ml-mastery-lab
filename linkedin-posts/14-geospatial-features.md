A neighbor average that had already seen the block groups being scored looked like a large win. It was not a holdout.

I froze five California cities and a half-degree cell split before fitting a ridge model on the 1990 census block groups. Development RMSE then picked haversine distance to those cities. On the held-out cells, which did not choose the recipe, that model had RMSE 0.6079330366056027. Predicting the training mean scored 0.9652129570335812. Raw latitude and longitude scored lower on those same cells, and the cell-cluster interval included zero. I did not switch after seeing that.

The clever feature failed the honest test. A mean of the 15 nearest other training block groups lost to a model with no location. Building that mean before the split scored 0.39365233020049006, and that interval excluded zero. A random split of the same size is a different question: almost every random row still sits in a cell the model was fit on.

If the rows you are about to score helped build the neighbor average, the number is not a forecast.

https://github.com/Manzoor25/ml-mastery-lab/tree/main/02-feature-engineering/06-geospatial-features

#MachineLearning #DataScience #FeatureEngineering #Geospatial #Python
