import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
import statistics
import scipy as sc
import joblib

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline

from sklearn.compose import make_column_selector as selector, ColumnTransformer
from sklearn import model_selection, preprocessing, metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,auc,RocCurveDisplay,fbeta_score, make_scorer, ConfusionMatrixDisplay
from sklearn.base import ClassifierMixin

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from lightgbm import LGBMClassifier

# Define data_path where cleaned data are stored
data_path = 'C:/Users/Melanie/Desktop/Formation_DS/P7/Cleaned_data/'

# Select a subset (10%) of initial dataset to select the best model (for performance)
df_features = pd.read_csv(data_path+"cleaned_data_most_important_features.csv").set_index('SK_ID_CURR')
df_target = pd.read_csv(data_path+"cleaned_data_most_important_target.csv").set_index('SK_ID_CURR')
df_full = pd.merge(df_features, df_target, left_index=True, right_index=True)
sample_dataset = df_full.sample(frac=0.1)

# Without sampling the dataset
sample_dataset = df_full

# Calculate scale_pos_weight:
scale_pos_weight = (sample_dataset['TARGET']==0).sum()/(sample_dataset['TARGET']==1).sum()
print(f"scale_pos_weight = {scale_pos_weight}")
# scale_pos_weight = number of negative samples / number of positive samples
# Here scale_pos_weight = 10.935706084959817

# Separate features and target
sample_dataset_features = sample_dataset.drop(columns=['TARGET'])
sample_dataset_target = sample_dataset['TARGET']

X = sample_dataset_features
y = sample_dataset_target

#-- Constants
# Hyperparameters ranges for each tested model
HYPERPARAMETERS = {'logreg': {"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}, 
                   'knn' : {"classifier__weights": ['uniform','distance'],"classifier__metric": ['minkowski','euclidean','manhattan'], "classifier__leaf_size": [1,5,10,15,20,30,40,50],"classifier__n_neighbors": [2,5,10,12,15,17,20,25,30],"classifier__p": [1, 2]},
                   'xgb' : {"classifier__learning_rate": [0.01, 0.1, 0.25, 0.5],"classifier__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],"classifier__min_child_weight": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],"classifier__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],"classifier__max_depth": [3, 5, 8, 11, 14, 16, 20],},
                   'xgb_weight' : {"classifier__learning_rate": [0.01, 0.1, 0.25, 0.5],"classifier__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],"classifier__min_child_weight": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],"classifier__colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],"classifier__max_depth": [3, 5, 8, 11, 14, 16, 20],"classifier__scale_pos_weight": [scale_pos_weight]},
                   'lgbm_smote' : {"classifier__objective" : ['binary'],"classifier__learning_rate": [0.01, 0.1, 0.25, 0.5],"classifier__boosting_type": ['gbdt', 'dart', 'goss'],"classifier__sub_feature": [0, 0.25, 0.5, 0.75, 1],"classifier__num_leaves": [20, 50, 100, 200, 300],"classifier__max_depth": [5, 50, 100, 150, 200],"classifier__min_data": [10, 20, 50, 70, 100],},
                   'lgbm_unbalance' : {"classifier__objective" : ['binary'],"classifier__learning_rate": [0.01, 0.1, 0.25, 0.5],"classifier__boosting_type": ['gbdt', 'dart', 'goss'],"classifier__sub_feature": [0, 0.25, 0.5, 0.75, 1],"classifier__num_leaves": [20, 50, 100, 200, 300],"classifier__max_depth": [5, 50, 100, 150, 200],"classifier__min_data": [10, 20, 50, 70, 100],"classifier__is_unbalance": [True]},
                   'lgbm_weight' : {"classifier__objective" : ['binary'],"classifier__learning_rate": [0.01, 0.1, 0.25, 0.5],"classifier__boosting_type": ['gbdt', 'dart', 'goss'],"classifier__sub_feature": [0, 0.25, 0.5, 0.75, 1],"classifier__num_leaves": [20, 50, 100, 200, 300],"classifier__max_depth": [5, 50, 100, 150, 200],"classifier__min_data": [10, 20, 50, 70, 100],"classifier__scale_pos_weight": [scale_pos_weight]},
                   }

# Classifiers
CLASSIFIERS = {'logreg': LogisticRegression(random_state=11, max_iter=1000),
               'knn':KNeighborsClassifier(),
               'xgb':xgb.XGBClassifier(),
               'xgb_weight':xgb.XGBClassifier(),
               'lgbm_smote':LGBMClassifier(),
               'lgbm_unbalance':LGBMClassifier(),
               'lgbm_weight':LGBMClassifier()}

# Target names (0 = the loan is repaid, 1 = the loan is delayed or not repayed)
TARGET_NAMES = ['repaid', 'delay']

# Plot confusion matrix from classifier including oversampling of training data
def plot_confusion_matrix_oversampling(classifier, x_train, y_train, x_test, y_test, use_oversampling=True):
    """From a classifier and associated x_train, y_train, x_test and y_test obtained after splitting original data into training and testing sets,
    plot the corresponding confusion matrix.

    Args:
        classifier: the classifier (could be a pipeline), not fitted
        x_train: training X data (unbalanced data)
        y_train: training y data (unbalanced data)
        x_test: testing X data
        y_test: testing y data
        use_oversampling: bool, determine if oversampling should be taken into account. Default: True.

    Returns:
        plot: a confusion matrix that takes into account the unbalanced training data by oversampling with smote
    """
    
    if use_oversampling:
        oversample = SMOTE()
        X_sm_train, y_sm_train = oversample.fit_resample(x_train, y_train)
        classifier.fit(X_sm_train, y_sm_train)
    else:
        classifier.fit(x_train, y_train)
        
    confusion_matrix_plot = ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test, normalize='true', cmap = 'Blues')
    
    return confusion_matrix_plot


# Cross validation function with metrics
def cross_validation(model, _X, _y, _cv=5):
    '''Function to perform k Folds Cross-Validation
    Parameters
    ----------
    model: Python Class, default=None
            This is the machine learning algorithm to be used for training.
    _X: array
        This is the matrix of features.
    _y: array
        This is the target variable.
    _cv: int, default=5
        Determines the number of folds for cross-validation.
    Returns
    -------
    The function returns a dictionary containing the metrics 'accuracy', 'precision',
    'recall', 'f1' for both training set and validation set.
    '''
    
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                            X=_X,
                            y=_y,
                            cv=_cv,
                            scoring=_scoring,
                            return_train_score=True)
    model_name = str(model)
    result_dict = {"index":model_name,
        "Training Accuracy scores": results['train_accuracy'],
            "Mean Training Accuracy": results['train_accuracy'].mean()*100,
            "Training Precision scores": results['train_precision'],
            "Mean Training Precision": results['train_precision'].mean(),
            "Training Recall scores": results['train_recall'],
            "Mean Training Recall": results['train_recall'].mean(),
            "Training F1 scores": results['train_f1'],
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Validation Accuracy scores": results['test_accuracy'],
            "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
            "Validation Precision scores": results['test_precision'],
            "Mean Validation Precision": results['test_precision'].mean(),
            "Validation Recall scores": results['test_recall'],
            "Mean Validation Recall": results['test_recall'].mean(),
            "Validation F1 scores": results['test_f1'],
            "Mean Validation F1 Score": results['test_f1'].mean()}

    result_df = pd.DataFrame.from_dict(result_dict)

    return result_df
    

def custom_cost_function(y_test, y_pred):
    """
    Custom cost function that penalizes false negatives (weight: 10x higher than false positives).
    Args:
        y_test: the true target values
        y_pred: predicted target values obtained after modelling
    Returns:
        Customized cost than penalizes more the false negatives values
    """
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Coefficient to calculate cost: make false negative 10 times more costly than false positives
    tp_value = 0 # default (no cost): could be changed if needed
    tn_value = 0 # default (no cost): could be changed if needed
    fp_value = 1 # low cost for false positives
    fn_value = 10 # penalized false negatives value (arbitratry value)
    
    # Calculate cost
    cost = fp*fp_value + fn*fn_value + tn*tn_value + tp*tp_value
    
    # Normalize cost according to the numbre of FP and FN
    #NB: add 1 in case there is no fp or fn at all
    cost = cost/(fp+fn+1)
    
    return cost

# For prediction, we also have to optimize the threshold used to determine the class (from probability to class)
custom_scorer = make_scorer(custom_cost_function, greater_is_better=False)


def model_hyperparameter_tuning(model_name:str, scoring:str, X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array, X:pd.DataFrame, y:pd.DataFrame, splits:int=6, use_oversampling:bool=True, scoring_name:str="roc_auc"):
    """From the model specified in model_name and the scoring function (for example roc_auc or custom_scorer), determine the best hyperparameter using randomizedsearchCV function
    from the specified X_train, y_train, X_test and y_test values.
    Note that the imbalanced-learn pipeline is used and imbalanced data are taken into account using SMOTE oversampling.

    Args:
        model_name (str): the model name (from keys of dicts used to store hyperparameters and classifiers values)
        scoring (str): desired type of scoring (example: roc_auc or custom_scorer function here).
        X_train (np.array): training set of features
        y_train (np.array): training set of targets
        X_test (np.array): testing set of features
        y_test (np.array): testing set of targets
        X (pd.DataFrame): dataframe containing features values with one feature per column and IDs as index
        y (pd.DataFrame): dataframe of one column containing target values and IDs as index
        splits (int): number of splits to consider during cross validation. Default 6
        use_oversampling (bool): determines if oversampling of training data should be used (SMOTE). Default: True
        scoring_name (str): type of scoring. Default "roc_auc"
    """

    # Define imbalanced pipeline with the right classifier for hyperparameter tuning
    if use_oversampling:
        pipeline = imbpipeline(
            steps=[
                ["smote", SMOTE(random_state=11)],
                ["scaler", StandardScaler()],
                ["classifier", CLASSIFIERS[model_name]],
            ]
        )
    else:
        pipeline = Pipeline([("scaler", StandardScaler()), 
                            ("classifier", CLASSIFIERS[model_name])])
    
    stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=11)
    param_grid = HYPERPARAMETERS[model_name]

    # Hyperparameter tuning using defined scoring function
    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        scoring=scoring,
        cv=stratified_kfold,
        n_jobs=-1,
        refit = True,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)
    
    cv_score = grid_search.best_score_
    test_score = grid_search.score(X_test, y_test)
    
    best_param = grid_search.best_params_
    
    # Run selected model with best hyperparameters
    classifier = select_classifier(model_name, best_param)
    
    # Run cross validation with 5 folds and store the mean recall for model comparison. Also exports the obtained model in joblib format
    cross_validation_results = run_cross_validation(model_name, classifier, X_train, y_train, X, y, splits, use_oversampling, scoring_name)
    
    return {"model": model_name, "best_param": best_param, "cv_score":cv_score, "test_score":test_score, 
            "mean_train_accuracy": cross_validation_results["cv_model_mean_train_accuracy"], 
            "mean_train_precision": cross_validation_results["cv_model_mean_train_precision"],
            "mean_train_recall": cross_validation_results["cv_model_mean_train_recall"],
            "mean_train_f1_score": cross_validation_results["cv_model_mean_train_f1_score"],
            "mean_val_accuracy": cross_validation_results["cv_model_mean_val_accuracy"],
            "mean_val_precision": cross_validation_results["cv_model_mean_val_precision"],
            "mean_val_recall": cross_validation_results["cv_model_mean_val_recall"],
            "mean_val_f1_score": cross_validation_results["cv_model_mean_val_f1_score"],
            "classifier": pipeline["classifier"]
            }


def select_classifier(model_name:str, best_param):
    """Select the right classifier according to model name and obtained tuned hyperparameters.

    Args:
        model_name (str): the model name
        best_param (): best parameters obtained for the selected model after running grid_search.best_params_
        
    Returns: corresponding classifier
    """
    
    if model_name == "logreg":
        classifier = LogisticRegression(random_state=11, max_iter=1000, C=best_param["classifier__C"])
    elif model_name == "knn":
        classifier = KNeighborsClassifier(weights=best_param['classifier__weights'], 
                metric=best_param['classifier__metric'], 
                leaf_size=best_param['classifier__leaf_size'], 
                n_neighbors=best_param['classifier__n_neighbors'],
                p=best_param['classifier__p'])
    elif model_name == "xgb":
        classifier = xgb.XGBClassifier(learning_rate=best_param['classifier__learning_rate'], 
                subsample=best_param['classifier__subsample'], 
                min_child_weight=best_param['classifier__min_child_weight'], 
                colsample_bytree=best_param['classifier__colsample_bytree'],
                max_depth=best_param['classifier__max_depth'])
    elif model_name == "xgb_weight":
        classifier = xgb.XGBClassifier(learning_rate=best_param['classifier__learning_rate'], 
                subsample=best_param['classifier__subsample'], 
                min_child_weight=best_param['classifier__min_child_weight'], 
                colsample_bytree=best_param['classifier__colsample_bytree'],
                max_depth=best_param['classifier__max_depth'],
                scale_pos_weight=best_param['classifier__scale_pos_weight'])
    elif model_name == "lgbm_smote":
        classifier = LGBMClassifier(objective=best_param['classifier__objective'],
                learning_rate=best_param['classifier__learning_rate'], 
                boosting_type=best_param['classifier__boosting_type'], 
                sub_feature=best_param['classifier__sub_feature'], 
                num_leaves=best_param['classifier__num_leaves'],
                max_depth=best_param['classifier__max_depth'],
                min_data=best_param['classifier__min_data'])
    elif model_name == "lgbm_unbalance":
        classifier = LGBMClassifier(objective=best_param['classifier__objective'],
                learning_rate=best_param['classifier__learning_rate'], 
                boosting_type=best_param['classifier__boosting_type'], 
                sub_feature=best_param['classifier__sub_feature'], 
                num_leaves=best_param['classifier__num_leaves'],
                max_depth=best_param['classifier__max_depth'],
                min_data=best_param['classifier__min_data'],
                is_unbalance=best_param['classifier__is_unbalance'])
    elif model_name == "lgbm_weight":
        classifier = LGBMClassifier(objective=best_param['classifier__objective'],
                learning_rate=best_param['classifier__learning_rate'], 
                boosting_type=best_param['classifier__boosting_type'], 
                sub_feature=best_param['classifier__sub_feature'], 
                num_leaves=best_param['classifier__num_leaves'],
                max_depth=best_param['classifier__max_depth'],
                min_data=best_param['classifier__min_data'],
                scale_pos_weight=best_param['classifier__scale_pos_weight'])
    
    return classifier


def run_cross_validation(model_name:str, classifier:ClassifierMixin, X_train:np.array, y_train:np.array, X:pd.DataFrame, y:pd.DataFrame, splits:int=6, use_oversampling:bool=True, scoring:str = "roc_auc"):
    """Run k-fold cross validation from a classifier that have optimized hyperparameters to obtain cross validation metrics.

    Args:
        model_name (str): the model name (from keys of dicts used to store hyperparameters and classifiers values)
        classifier (ClassifierMixin): considered classifier, with optimized hyperparameters
        X_train (np.array): features values in the training set
        y_train (np.array): target values in the training set
        X (pd.DataFrame): dataframe containing features as columns and id as index
        y (pd.DataFrame): dataframe of one column containing the target values and id as index
        split (int): number of splits for k-fold cross-validation. Default 6
        use_oversampling (bool): determines if we should consider an oversampling method (True) or not. Default True
        scoring (str): type of scoring (either "roc_auc" or "custom_scorer" here). Default "roc_auc"
        
    returns: a dict containing metrics with metric name as key and its mean value as value
    """
    
    if use_oversampling:
        pipeline = imbpipeline(
            steps=[
                ["smote", SMOTE(random_state=11)],
                ["scaler", StandardScaler()],
                [
                    "classifier",
                    classifier,
                ],
            ]
        )
    else: 
        pipeline = Pipeline([("scaler", StandardScaler()), 
                            ("classifier", classifier)])
    
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, f'{model_name}_pipeline_{scoring}.joblib')

    cv_model = cross_validation(pipeline,X,y,_cv=splits)
    return {"cv_model_mean_train_accuracy": cv_model["Mean Training Accuracy"].mean(),
    "cv_model_mean_train_precision": cv_model["Mean Training Precision"].mean(),
    "cv_model_mean_train_recall": cv_model["Mean Training Recall"].mean(),
    "cv_model_mean_train_f1_score": cv_model["Mean Training F1 Score"].mean(),
    "cv_model_mean_val_accuracy": cv_model["Mean Validation Accuracy"].mean(),
    "cv_model_mean_val_precision": cv_model["Mean Validation Precision"].mean(),
    "cv_model_mean_val_recall": cv_model["Mean Validation Recall"].mean(),
    "cv_model_mean_val_f1_score": cv_model["Mean Validation F1 Score"].mean()}
    
    
def print_classification_report(model_name, classifier, sampling, target_names, X_train, y_train, X_test, y_test, use_oversampling=True):
    """From a specific classifier with its optimized hyperparameters corresponding to a classification model, print the classification report
    using training and testing sets from original data, assuming that the data are imbalanced. The training data are here oversampled or undersampled
    using the provided sampling method.

    Args:
        model_name (string): name of the tested model
        classifier (_type_): the classifier with the right hyperparameters
        sampling (_type_): oversampling or undersampling method used for imbalanced data
        target_names (list): list of two strings corresponding to the two target classes used for classification
        X_train (array): training set of features
        y_train (array): training set of targets
        X_test (array): testing set of features
        y_test (array): testing set of targets
        use_oversampling: bool, default True. Determines if oversampling should be used for training data.
    returns: the classification report as a string output
    
    """
    if use_oversampling:
        X_sm, y_sm = sampling.fit_resample(X_train, y_train)
        classifier.fit(X_sm, y_sm)
    else:
        classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    classification_report_model = classification_report(y_test, y_pred, target_names=target_names)
    print(f"\nClassification report for {model_name} model:", classification_report_model)
    return classification_report_model


def run_dummy_classifier(X:np.array, y:np.array, cm_name:str, test_size:float=0.30, use_oversampling:bool=True)->dict:
    """Run dummy classifier for classification using "stratified" strategy to reflect the potential data unbalance.

    Args:
        X (np.array): features (one column per feature, one row per client id)
        y (np.array): target value (one row per client id)
        cm_name (str): png file name corresponding to the confusion matrix
        test_size (float, optional): Test set proportion. Defaults to 0.30.
        use_oversampling (bool, optional): determines if oversampling with SMOTE is needed. Defaults to True.

    Returns:
        dict: metrics with metric name as key and its mean value as value
    """
    
    X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(X, y, test_size = test_size)
    clf_dummy = DummyClassifier(random_state=42, strategy = "stratified")

    if use_oversampling:
        dummy_results = run_cross_validation('dummy', classifier=clf_dummy, X_train=X_train_dummy, y_train=y_train_dummy,X=X, y=y,use_oversampling=True, scoring = "none")
        dummy_confusion_matrix = plot_confusion_matrix_oversampling(clf_dummy, X_train_dummy, y_train_dummy, X_test_dummy, y_test_dummy, use_oversampling=True)
        print_classification_report('dummy', clf_dummy, oversample, TARGET_NAMES, X_train_dummy, y_train_dummy, X_test_dummy, y_test_dummy, use_oversampling=True)
    
    else:
        dummy_results = run_cross_validation('dummy', classifier=clf_dummy, X_train=X_train_dummy, y_train=y_train_dummy,X=X, y=y,use_oversampling=False, scoring = "none")
        dummy_confusion_matrix = plot_confusion_matrix_oversampling(clf_dummy, X_train_dummy, y_train_dummy, X_test_dummy, y_test_dummy, use_oversampling=False)
        print_classification_report('dummy', clf_dummy, oversample, TARGET_NAMES, X_train_dummy, y_train_dummy, X_test_dummy, y_test_dummy, use_oversampling=False)
    
    dummy_confusion_matrix.figure_.savefig(f"{cm_name}.png")
    
    return dummy_results


def run_model_with_hyperparameter_tuning(scoring, model_name:str, X:np.array, y:np.array, test_size:float=0.3, scoring_name='roc_auc', use_oversampling:bool=True)->dict:
    """Run the obtained model with the tuning of hyperparameters to obtain classification performance metrics and confusion matrix.

    Args:
        scoring: scoring method (here 'roc_auc' or custom_scorer for example)
        model_name (str): model name
        X (np.array): features with one column per feature and one row per client id
        y (np.array): target with one row per client id
        test_size (float, optional): test set proportion. Defaults to 0.3.
        scoring (str, optional): Scoring method. Defaults to 'roc_auc'.
        use_oversampling (bool, optional): determines if oversampling with SMOTE should be used. Default to True

    Returns:
        dict: dict containing model metrics with one key per metric name
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=11)
    
    if use_oversampling:
        model_result = model_hyperparameter_tuning(model_name, scoring, X_train, y_train, X_test, y_test, X, y, scoring_name = scoring_name, use_oversampling=True)
        model_classifier = joblib.load(f"{model_name}_pipeline_{scoring_name}.joblib")
        print_classification_report(model_name, model_classifier, oversample, TARGET_NAMES, X_train, y_train, X_test, y_test, use_oversampling=True)
        model_confusion_matrix = plot_confusion_matrix_oversampling(model_classifier, X_train, y_train, X_test, y_test, use_oversampling=True)
        model_confusion_matrix.figure_.savefig(f"{model_name}_cm_{scoring_name}_oversampling.png")
    else:
        model_result = model_hyperparameter_tuning(model_name, scoring, X_train, y_train, X_test, y_test, X, y, scoring_name = scoring_name, use_oversampling=False)
        model_classifier = joblib.load(f"{model_name}_pipeline_{scoring_name}.joblib")
        print_classification_report(model_name, model_classifier, oversample, TARGET_NAMES, X_train, y_train, X_test, y_test, use_oversampling=False)
        model_confusion_matrix = plot_confusion_matrix_oversampling(model_classifier, X_train, y_train, X_test, y_test, use_oversampling=False)
        model_confusion_matrix.figure_.savefig(f"{model_name}_cm_{scoring_name}_no_oversampling.png")
    
    return model_result


# Using SMOTE to balance the training data if requested
oversample = SMOTE()

# Main result for comparison storing in dict for the two types of scoring (roc auc and custom scorer)
mean_recall_cv_dict_auc = {}
mean_recall_cv_dict_custom = {}
results_dict = {}

#----- Dummy classifier (baseline results for comparison)
print("#----- Dummy classifier with SMOTE oversampling -----#")
dummy_results_smote = run_dummy_classifier(X, y, 'dummy_cm_smote', 0.3, True)

print("#----- Dummy classifier without oversampling -----#")
dummy_results_without_smote = run_dummy_classifier(X, y, 'dummy_cm_without_smote', 0.3, False)

#----- Logistic regression
print("#----- Logistic regression model -----#")
logreg_auc_no_oversampling = run_model_with_hyperparameter_tuning(scoring='roc_auc',model_name='logreg',X=X,y=y,test_size=0.3,scoring_name="roc_auc",use_oversampling=False)
logreg_auc_oversampling = run_model_with_hyperparameter_tuning(scoring='roc_auc',model_name='logreg',X=X,y=y,test_size=0.3,scoring_name="roc_auc",use_oversampling=True)

logreg_custom_no_oversampling = run_model_with_hyperparameter_tuning(scoring=custom_scorer,model_name='logreg',X=X,y=y,test_size=0.3,scoring_name="custom_scorer",use_oversampling=False)
logreg_custom_oversampling = run_model_with_hyperparameter_tuning(scoring=custom_scorer,model_name='logreg',X=X,y=y,test_size=0.3,scoring_name="custom_scorer",use_oversampling=True)

#----- kNN classifier
print("#----- kNN classifier model -----#")
knn_auc_oversampling = run_model_with_hyperparameter_tuning(scoring='roc_auc',model_name='knn',X=X,y=y,test_size=0.3,scoring_name="roc_auc",use_oversampling=True)
knn_custom_oversampling = run_model_with_hyperparameter_tuning(scoring=custom_scorer,model_name='knn',X=X,y=y,test_size=0.3,scoring_name="custom_scorer",use_oversampling=True)

#----- XGBoost
# See https://blog.dataiku.com/narrowing-the-search-which-hyperparameters-really-matter to determine which hyperparameters are more important than others
# for this model

print("#----- XGBoost model with oversampling (SMOTE) -----#")
xgb_auc_oversampling = run_model_with_hyperparameter_tuning(scoring='roc_auc',model_name='xgb',X=X,y=y,test_size=0.3,scoring_name="roc_auc",use_oversampling=True)
xgb_custom_oversampling = run_model_with_hyperparameter_tuning(scoring=custom_scorer,model_name='xgb',X=X,y=y,test_size=0.3,scoring_name="custom_scorer",use_oversampling=True)

print("#----- XGBoost model with scale_pos_weight-----#")
xgb_weight_auc = run_model_with_hyperparameter_tuning(scoring='roc_auc',model_name='xgb_weight',X=X,y=y,test_size=0.3,scoring_name="roc_auc",use_oversampling=False)
xgb_weight_custom = run_model_with_hyperparameter_tuning(scoring=custom_scorer,model_name='xgb_weight',X=X,y=y,test_size=0.3,scoring_name="custom_scorer",use_oversampling=False)

#----- LightGBM

print("#----- LightGBM model with oversampling (SMOTE) -----#")
lgbm_auc_oversampling = run_model_with_hyperparameter_tuning(scoring='roc_auc',model_name='lgbm_smote',X=X,y=y,test_size=0.3,scoring_name="roc_auc",use_oversampling=True)
lgbm_custom_oversampling = run_model_with_hyperparameter_tuning(scoring=custom_scorer,model_name='lgbm_smote',X=X,y=y,test_size=0.3,scoring_name="custom_scorer",use_oversampling=True)

print("#----- LightGBM model with is_unbalance=True -----#")
lgbm_auc_unbalance = run_model_with_hyperparameter_tuning(scoring='roc_auc',model_name='lgbm_unbalance',X=X,y=y,test_size=0.3,scoring_name="roc_auc",use_oversampling=False)
lgbm_custom_unbalance = run_model_with_hyperparameter_tuning(scoring=custom_scorer,model_name='lgbm_unbalance',X=X,y=y,test_size=0.3,scoring_name="custom_scorer",use_oversampling=False)

print("#----- LightGBM model with scale_pos_weight -----#")
lgbm_auc_weight = run_model_with_hyperparameter_tuning(scoring='roc_auc',model_name='lgbm_weight',X=X,y=y,test_size=0.3,scoring_name="roc_auc",use_oversampling=False)
lgbm_custom_weight = run_model_with_hyperparameter_tuning(scoring=custom_scorer,model_name='lgbm_weight',X=X,y=y,test_size=0.3,scoring_name="custom_scorer",use_oversampling=False)

# Dataframe containing interesting metrics values for each model among those which display the best confusion matrix
metrics_dataframe_custom = pd.DataFrame.from_dict([dummy_results_smote, logreg_custom_oversampling, knn_custom_oversampling, xgb_custom_oversampling, xgb_weight_custom, lgbm_custom_oversampling, lgbm_custom_unbalance, lgbm_custom_weight])
metrics_dataframe_custom.to_csv("metrics_comparison_custom_scorer.csv")

metrics_dataframe_auc = pd.DataFrame.from_dict([dummy_results_smote, logreg_auc_oversampling, knn_auc_oversampling, xgb_auc_oversampling, xgb_weight_auc, lgbm_auc_oversampling, lgbm_auc_unbalance, lgbm_auc_weight])
metrics_dataframe_auc.to_csv("metrics_comparison_roc_auc.csv")

