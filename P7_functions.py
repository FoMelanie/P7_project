import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def calculate_nan_percentage_per_column(df):
    """
    Calculate the NaN percentage per column of an input dataframe (df) and show it in a new dataframe.
    :param df: pandas dataframe
    :return: a dataframe containing the column name and the corresponding NaN percentage in a second column.
    The dataframe is sorted according to the NaN percentage in a descending order.
    """
    column_list = df.columns.values.tolist()
    percentage_value = (1 - round(df[column_list].count() / len(df), 2)) * 100
    final_df = pd.DataFrame(list(zip(column_list, percentage_value)), columns=["Columns", "Percentage of NaN"])
    return final_df.sort_values(by=['Percentage of NaN'], ascending=False)


def remove_outliers_with_iqr_method(df, column_name, type_of_outlier_to_remove):
    """
    Using the inter quantile ratio method, outliers are removed by this function. Namely, the Q3-Q1 is calculated as the IQR and
    an upper and lower limits are defined as thresholds, respectively by Q3 + 1.5 * IQR and Q1 - 1.5 * IQR. Values that are upper
    the upper limit or lower than the lower limits are considered outliers and removed from the analysis.
    :param df: input pandas dataframe
    :param column_name: string, column name from which we want to remove outliers.
    :param type_of_outlier_to_remove: string, either "lower", "upper" or "lower_upper". Defines if we want to return a
    pandas dataframe only without lower or upper outliers or we want to return a pandas dataframe without upper and lower outliers.
    :return: a pandas dataframe without rows that have a column_name value considered as outliers.
    """

    # Calculate percentiles and corresponding IQR
    percentile25 = df[column_name].quantile(0.25)
    percentile75 = df[column_name].quantile(0.75)
    iqr = percentile75 - percentile25

    # Define upper and lower thresholds
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr

    # Eliminate outliers
    if type_of_outlier_to_remove == "lower":
        final_df = df[df[column_name] > lower_limit]
    elif type_of_outlier_to_remove == "upper":
        final_df = df[df[column_name] < upper_limit]
    elif type_of_outlier_to_remove == "lower_upper":
        df_without_low_outliers = df[df[column_name] > lower_limit]
        final_df = df_without_low_outliers[df_without_low_outliers[column_name] < upper_limit]
    else:
        print("\nThe argument type_of_outlier_to_remove should be either \"lower\", \"upper\" or \"lower_upper\".\n")

    return final_df


def save_model_scores(model_name: str, model_variable, y_test, y_predicted, X, y):
    """
    From a trained regression model, calculates the basic metrics used notably to compare model performances in the
    form of a pandas dataframe. The values are rounded to 3 decimals and printed in the console for information.
    :param model_name: string, defines the model name as it will appear in the final dataframe
    :param model_variable: variable that contains the trained model
    :param y_test: test set of the target
    :param y_predicted: predicted values of the target obtained using the model
    :param X: original features, without split into train and test
    :param y: original target, without split into train and test
    :return: a dataframe containing the following regression metrics: mean absolute error, mean squared error,
    mean root squared error, R² score, cross validation score.
    """
    # Basic regression metrics
    MAE = mean_absolute_error(y_test, y_predicted)
    MSE = mean_squared_error(y_test, y_predicted, squared=True)
    RMSE = mean_squared_error(y_test, y_predicted, squared=False)
    R2_score = r2_score(y_test, y_predicted)

    print("Mean squared error: %.2f" % MSE)
    print("Root mean squared error: %.2f" % RMSE)
    print("Mean absolute error: %.2f" % MAE)
    print("Coefficient of determination: %.2f" % R2_score)

    # Cross validation results
    cv_result = cross_validate(model_variable, X, y, cv=10)
    cv_result

    scores = cv_result["test_score"]
    print(
        "The mean cross-validation accuracy is: "
        f"{scores.mean():.3f} +/- {scores.std():.3f}"
    )

    # Save results in a dict
    score_dict = {'MAE': round(MAE, 3),
                  'MSE': round(MSE, 3),
                  'RMSE': round(RMSE, 3),
                  'R2_Score': round(R2_score, 3),
                  'Cross_Validation': f"{scores.mean():.3f} +/- {scores.std():.3f}"}

    # Convert dict into dataframe
    df = pd.DataFrame.from_dict(score_dict, orient='index')

    # Set model name as df column name
    df = df.rename(columns={0: model_name})

    return df


def perf_measure(y_true, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_true[i]==y_pred[i]==1:
           TP += 1
        elif y_pred[i]==1 and y_true[i]!=y_pred[i]:
           FP += 1
        elif y_true[i]==y_pred[i]==0:
           TN += 1
        elif y_pred[i]==0 and y_true[i]!=y_pred[i]:
           FN += 1
           
    print(f"TP : {TP}, FP : {FP}, TN : {TN}, FN : {FN}")
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print(f"Sensitivity (true positive rate) : {TPR}")
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    print(f"Specificity (true negative rate) : {TNR}")
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print(f"Precision (positive predictive value) : {PPV}")

    return(TP, FP, TN, FN, TPR, TNR, PPV)


def generate_model_report(y_actual, y_predicted):
    print("Accuracy = " , accuracy_score(y_actual, y_predicted))
    print("Precision = " ,precision_score(y_actual, y_predicted))
    print("Recall = " ,recall_score(y_actual, y_predicted))
    print("F1 Score = " ,f1_score(y_actual, y_predicted))
    pass


def generate_auc_roc_curve(clf, X_test, Y_test):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(Y_test,  y_pred_proba)
    auc = roc_auc_score(Y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="AUC ROC Curve with Area Under the curve ="+str(auc))
    plt.legend(loc=4)
    plt.show()
    pass

