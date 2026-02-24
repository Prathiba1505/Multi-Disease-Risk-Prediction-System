import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import joblib

def load_csv(path):
    return pd.read_csv(path)

def splitdata(x,y):
    return train_test_split(x,y,test_size=0.2,random_state=42)

def trainmodel(x_train,y_train):
    model=RandomForestClassifier(n_estimators=200,max_depth=5,random_state=42)
    model.fit(x_train,y_train)
    return model
    
def evaluate(model,x_test,y_test):
    prediction=model.predict(x_test)
    print("Prediction:")
    print(prediction)

    accuracy=accuracy_score(prediction,y_test)
    print("Accuracy:")
    print(accuracy)

    print("Confusion Matrix")
    print(confusion_matrix(y_test,prediction))

    print("Classification report")
    print(classification_report(y_test,prediction))

def main():
    df=load_csv("../data/diabetes.csv")

    print("First 5 rows:")
    print(df.head())

    print("Shape of Dataset")
    print(df.shape)

    print("Missing values")
    print(df.isnull().sum())

    x=df.drop("Outcome",axis=1)
    y=df["Outcome"]

    x_train,x_test,y_train,y_test=splitdata(x,y)

    model=trainmodel(x_train,y_train)

    evaluate(model,x_test,y_test)

    joblib.dump(model,"../diabetes_model.pkl")


if __name__=="__main__":
    main()
