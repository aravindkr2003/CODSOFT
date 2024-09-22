from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise import accuracy
from surprise.model_selection import train_test_split

data = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(data, test_size=0.25)

sim_options = {
    'name': 'cosine',  
    'user_based': True   filtering
}

algo = KNNBasic(sim_options=sim_options)

algo.fit(trainset)

predictions = algo.test(testset)

accuracy.rmse(predictions)

user_id = str(196)
item_id = str(302)
predicted_rating = algo.predict(user_id, item_id)
print(f"Predicted rating for User {user_id} on Movie {item_id}: {predicted_rating.est}")