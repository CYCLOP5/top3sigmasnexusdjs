import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('train.csv')

le = LabelEncoder()
le.fit(data['Target_Status'])

model = joblib.load('best_model.pkl')

test_data = pd.read_csv('test.csv')  

X_test_submission = test_data.drop(columns=['ID'])  

X_test_submission['DrugType_HighStatus'] = X_test_submission['DRUGTYPE'].astype(str) + '_' + X_test_submission['Drug_high_status'].astype(str)
X_test_submission['DiseaseStatus_DrugStatus'] = X_test_submission['Disease_of_highest_status'].astype(str) + '_' + X_test_submission['Drug_Status'].astype(str)
X_test_submission['Unique_TargetID'] = X_test_submission['UNIPROID'].astype(str) + '_' + X_test_submission['TARGNAME'].astype(str) + '_' + X_test_submission['GENENAME'].astype(str)
X_test_submission['BioClass_Function'] = X_test_submission['BIOCLASS'].astype(str) + '_' + X_test_submission['FUNCTION'].astype(str)

X_test_submission = X_test_submission.drop(columns=['DRUGTYPE', 'Drug_high_status', 'Disease_of_highest_status', 'Drug_Status', 'UNIPROID', 'TARGNAME', 'GENENAME', 'BIOCLASS', 'FUNCTION'])

predictions_encoded = model.predict(X_test_submission)

predictions = le.inverse_transform(predictions_encoded)

submission_df = pd.DataFrame({
    'ID': test_data['ID'],  
    'Prediction': predictions
})

submission_df.to_csv('submission.csv', index=False)
print("Submission file created as 'submission.csv'")