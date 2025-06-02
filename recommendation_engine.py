from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
import pandas as pd


meetup_datset = pd.load_from_csv()

trainset, testset = train_test_split(meetup_datset, test_size=.2, random_state=42)

svd_recommender = SVD()
svd_recommender.fit(trainset)

svd_predictions = svd_recommender.test(testset)

accuracy.rmse(svd_predictions)