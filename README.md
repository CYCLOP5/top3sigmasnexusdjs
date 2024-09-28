## 1. `csvcreation.py`
**Purpose**: Generates the CSV file required for Kaggle submissions.  
**Description**: This script reads the model's predictions and formats them into a CSV file that follows Kaggle's submission guidelines.

---

## 2. `mainwithbestparamfromoptuna.py`
**Purpose**: Utilizes the best hyperparameters found from an Optuna study to train and evaluate the model.  
**Description**: After hyperparameter tuning, this script loads the best-performing parameters from the Optuna study and applies them to the model training process. It also evaluates model performance and generates predictions for submission.

---

## 3. `optunastudy.py`
**Purpose**: Conducts hyperparameter optimization using Optuna.  
**Description**: This script sets up an Optuna study to find the best hyperparameters for the model. It explores various configurations and selects the optimal set of parameters based on the study's objective function, typically maximizing the F1 score or minimizing loss.
### Workflow
1. **Parameter Tuning**: Run `optunastudy.py` to find the best hyperparameters.
2. **Model Training**: Use `mainwithbestparamfromoptuna.py` to apply the best parameters and generate predictions.
3. **Submission**: Format the output using `csvcreation.py` for easy Kaggle submission.
