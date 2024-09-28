import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv('/kaggle/input/nexus-by-djs-nsdc-ultraceuticals/train.csv')

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
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)]
)

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500)  
    max_depth = trial.suggest_categorical('max_depth', [None, 20, 30, 40])  
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)  
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)  
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])  

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,  
            random_state=42))
    ])

    skf = StratifiedKFold(n_splits=3)
    f1_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)  
    return f1_scores.mean()  

sampler = optuna.samplers.TPESampler(multivariate=True)  
pruner = optuna.pruners.MedianPruner()  

study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=500, n_jobs=-1)  

print(f"Best F1 Score: {study.best_value}")
print(f"Best Parameters: {study.best_params}")

best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=study.best_params['n_estimators'],
        max_depth=study.best_params['max_depth'],
        min_samples_split=study.best_params['min_samples_split'],
        min_samples_leaf=study.best_params['min_samples_leaf'],
        max_features=study.best_params['max_features'],
        n_jobs=-1,  
        random_state=42))
])

best_pipeline.fit(X_train, y_train)