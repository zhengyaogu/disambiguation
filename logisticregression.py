from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json

data = None
target = None
with open("trainingx.json") as t:
    data = json.load(t)
with open("trainingy.json") as m:
    target = json.load(m)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)

logisticReg = LogisticRegression()
logisticReg.fit(x_train, y_train)
predictions = logisticReg.predict(x_test)
score = logisticReg.score(x_test, y_test)
print("score:", score)
print("There were", len(x_train), "training examples and", len(x_test), "test examples")
#print(predictions)




