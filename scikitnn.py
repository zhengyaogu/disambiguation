from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import json

data = None
target = None
with open("trainingx.json") as t:
    data = json.load(t)
with open("trainingy.json") as m:
    target = json.load(m)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=0)
print("There are", len(x_train), "training examples and", len(x_test), "test examples")
clf = MLPClassifier(solver="adam", verbose=2, alpha=1e-5, hidden_layer_sizes=(1500), random_state=1, max_iter=40)
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
score = clf.score(x_test, y_test)
print("score:", score)