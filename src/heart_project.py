import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def load_csv(path):
    return pd.read_csv(path)

def splitdata(x,y):
    return train_test_split(x,y,test_size=0.2,random_state=42)

def trainmodel(x_train,y_train):
    # model=LogisticRegression(max_iter=5000)
    model=RandomForestClassifier(n_estimators=3000,max_depth=6,random_state=42)
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

    # scaler=StandardScaler()

    # x_train=scaler.fit_transform(x_train)
    # x_test=scaler.transform(x_test)

    model=trainmodel(x_train,y_train)

    evaluate(model,x_test,y_test)

    print("Enter the new patient details")

    new_patient = []

    numeric_limit={
        "age":(0,100),
        "trestbps": (80, 250),
        "chol": (100, 600),
        "thalach": (60, 250),
        "oldpeak": (0, 10)
    }

    allowed_values = {
    "sex": "[0 = Female, 1 = Male]",
    "cp": "[0,1,2,3]",
    "fbs": "[0 = No, 1 = Yes]",
    "restecg": "[0,1,2]",
    "exang": "[0 = No, 1 = Yes]",
    "slope": "[0,1,2]",
    "ca": "[0,1,2,3,4]",
    "thal": "[0,1,2,3]"
}

    for feature in x.columns:
        if feature in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
            value = int(input(f"Enter {feature} {allowed_values[feature]}: "))
            while True:
                if feature == "sex" and value in [0,1]:
                    break
                elif feature == "cp" and value in [0,1,2,3]:
                    break
                elif feature == "fbs" and value in [0,1]:
                    break
                elif feature == "restecg" and value in [0,1,2]:
                    break
                elif feature == "exang" and value in [0,1]:
                    break
                elif feature == "slope" and value in [0,1,2]:
                    break
                elif feature == "ca" and value in [0,1,2,3,4]:
                    break
                elif feature == "thal" and value in [0,1,2,3]:
                    break
                else:
                    print("Invalid input! Please enter a valid value.")
                    value = int(input(f"Enter {feature} {allowed_values[feature]}: "))
        else:
            min_value,max_value=numeric_limit[feature]
            value = float(input(f"Enter {feature} ({min_value} - {max_value}): "))
            while(value<min_value or value>max_value):
                print(f"Invalid input! Enter a value between {min_value} and {max_value}")
                value=float(input(f"Enter {feature} ({min_value} - {max_value}):"))

        new_patient.append(value)

    new_patient_df=pd.DataFrame([new_patient], columns=x.columns)
    # new_patient_scaled=scaler.transform(new_patient_df)

    # prediction=model.predict(new_patient_scaled)
    prediction=model.predict(new_patient_df)

    print("Heart Disease Prediction: ","Yes" if prediction[0]==1  else "No")

    # prob_no = model.predict_proba(new_patient_scaled)[0][0]*100
    # prob_yes = model.predict_proba(new_patient_scaled)[0][1]*100
    prob_no = model.predict_proba(new_patient_df)[0][0]*100
    prob_yes = model.predict_proba(new_patient_df)[0][1]*100
    print(f"Probability No: {prob_no:.2f}%, Yes: {prob_yes:.2f}%")

    if(prob_yes>=0 and prob_yes<40):
        print("Risk Level:Low risk")
    elif(prob_yes>=40 and prob_yes<60):
        print("Risk Level:Moderate risk")
    elif(prob_yes>=60 and prob_yes<80):
        print("Risk Level:High risk")
    else:
        print("Risk Level:Very high risk")

    prob=model.predict_proba(x_test)[:,1]
    print("AUC Score: ",roc_auc_score(y_test,prob))

if __name__=='__main__':
    main()