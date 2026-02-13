import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data={
    "age": [25, 45, 35, 50, 23, 40, 60, 48, 55, 30],
    "bmi": [22, 30, 27, 28, 21, 26, 31, 29, 33, 24],
    "glucose": [90, 180, 120, 200, 85, 110, 210, 160, 190, 95],
    "cholesterol": [180, 240, 200, 260, 170, 190, 280, 230, 250, 175],
    "exercise_hours": [5, 1, 3, 2, 6, 3, 0, 2, 1, 4],
    "disease": [0, 1, 0, 2, 0, 0, 2, 1, 2, 0]
}

df=pd.DataFrame(data)

x=df[['age','bmi','glucose','cholesterol','exercise_hours']]
y=df['disease']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)

model=LogisticRegression(max_iter=1000)

model.fit(x_train,y_train)

new_patient = [[28, 24, 100, 180, 5]]

prediction=model.predict(x_test)
prediction1=model.predict(new_patient)

accuracy=accuracy_score(y_test,prediction)
accuracy1=accuracy_score(y_test,prediction1)

print(prediction)
print(prediction1)
print(accuracy)
print(accuracy1)