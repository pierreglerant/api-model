import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
import joblib

def ingest_data(file_path:str) -> pd.DataFrame:
    return(pd.read_excel(file_path))

def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    df = df[['survived', 'pclass', 'sex', 'age']]
    df = df.dropna()
    df['sex'] = df['sex'].replace(to_replace=['male','female'],value=[0,1])
    return(df)

def train_model(df:pd.DataFrame) -> ClassifierMixin:
    model = KNeighborsClassifier()
    y=df['survived']
    X = df.drop('survived',axis=1)
    trainX,valX,trainY,valY = train_test_split(X,y,test_size = 0.2,random_state=2)
    model.fit(trainX,trainY)
    score = model.score(valX, valY)
    print(f'Model score : {score}')
    return(model)

if __name__ == "__main__":
    df = ingest_data("C:/Users/pierr/OneDrive/Documents/MLOPS/api model github/api-model/train/titanic.xls")
    df2 = clean_data(df)
    model2 = train_model(df2)
    joblib.dump(model2,filename="model_titanic.joblib")