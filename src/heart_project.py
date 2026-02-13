import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler

def load_csv(path):
    return pd.read_csv(path)

def splitdata(x,y):
    return train_test_split(x,y,test_size=0.2,random_state=42)

def trainmodel(x_train,y_train):
    model=LogisticRegression(max_iter=5000)
    model.fit(x_train,y_train)
    return model

def evaluate(model,x_test,y_test):
    prediction=model.predict(x_test)
    print("Prediction")
    print(prediction)

    print("Accuracy")
    print(accuracy_score(y_test,prediction))

    print("Confusion Matrix")
    print(confusion_matrix(y_test,prediction))

    print("Classification report")
    print(classification_report(y_test,prediction))

def main():
    df=load_csv("../data/heart.csv")

    print("First 5 rows")
    print(df.head())

    print("Data shape")
    print(df.shape)

    print("Data info")
    print(df.info())

    print("Statistical Summary")
    print(df.describe())

    x=df.drop("target",axis=1)
    y=df["target"]

    print("Feature shape")
    print(x.shape)
    print("Target shape")
    print(y.shape)

    x_train,x_test,y_train,y_test=splitdata(x,y)

    print("Training data shape")
    print(x_train.shape)
    print("Testing data shape")
    print(x_test.shape)

    scaler=StandardScaler()

    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    model=trainmodel(x_train,y_train)

    evaluate(model,x_test,y_test)

if __name__=='__main__':
    main()