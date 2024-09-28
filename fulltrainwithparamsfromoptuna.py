import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

data = pd.read_csv('train.csv')

X = data.drop(columns=['ID', 'Target_Status', 'TargetID', 'DRUGID', 'DRUGNAME', 'SEQUENCE', 'Accession Number'])
y = data['Target_Status']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X['DrugType_HighStatus'] = X['DRUGTYPE'].astype(str) + '_' + X['Drug_high_status'].astype(str)

X['DiseaseStatus_DrugStatus'] = X['Disease_of_highest_status'].astype(str) + '_' + X['Drug_Status'].astype(str)

X['Unique_TargetID'] = X['UNIPROID'].astype(str) + '_' + X['TARGNAME'].astype(str) + '_' + X['GENENAME'].astype(str)

X['BioClass_Function'] = X['BIOCLASS'].astype(str) + '_' + X['FUNCTION'].astype(str)

X = X.drop(columns=['DRUGTYPE', 'Drug_high_status', 'Disease_of_highest_status', 'Drug_Status', 'UNIPROID', 'TARGNAME', 'GENENAME', 'BIOCLASS', 'FUNCTION'])

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

best_params = {
    'n_estimators': 240,
    'max_depth': None,
    'min_samples_split':4,
    'min_samples_leaf': 1,
    'max_features': 'log2'
}

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        random_state=42,
        n_jobs=-1))

])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"F1 Score with Best Parameters: {f1}")

joblib.dump(pipeline, 'best_model_direct.pkl')
print("Model saved as 'best_model_direct.pkl'")