from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Dataset

movie_data = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(movie_data, test_size=.2, random_state=42)

svd_recommender = SVD()
svd_recommender.fit(trainset)

svd_predictions = svd_recommender.test(testset)

accuracy.rmse(svd_predictions)