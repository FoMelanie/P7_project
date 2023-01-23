import pandas as pd
import numpy as np
from typing import List
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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

# Define data_path where feature engineered data are stored
data_path = data_path = 'C:/Users/Melanie/Desktop/Formation_DS/P7/Cleaned_data/'

# Constants
NAN_PERCENTAGE_CLEANING = 30 # Percentage of NaN values among which a datframe column will be dropped 
CORRELATION_COEFFICIENT = 0.95 # Correlation coefficient to determine groups of highly correlated features

# Functions
def clean_dataset(df:pd.DataFrame)-> pd.DataFrame:
    """Cleans dataset to remove rows that contain NaN or infinite values.
    Args:
        df: pandas dataframe containing the values we want to clean
    Returns: the cleaned pandas dataframe
    """
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def remove_columns_according_to_nan_percentage(df:pd.DataFrame,nan_percentage:float=30)-> pd.DataFrame:
    """From a pandas dataframe and the indicated percentage of NaN values, remove the columns that contain more
    than this percentage of NaN values and return the filtered dataframe.

    Args:
        df (pd.DataFrame): The pandas dataframe that will be filtered.
        nan_percentage (float, optional): The percentage of NaN values among which the column is discarded. Defaults to 30.

    Returns:
        pd.DataFrame: The filtered pandas dataframe.
    """
    nan_data = calculate_nan_percentage_per_column(df)
    nan_columns = nan_data['Columns'][nan_data['Percentage of NaN']>nan_percentage].values.tolist()
    final_df = df.drop(columns=nan_columns)
    return final_df

# Import cleaned datasets obtained after feature engineering
previous_applications_cleaned = pd.read_csv(data_path+'previous_applications_cleaned.csv')
pos_cash_cleaned = pd.read_csv(data_path+'pos_cash_cleaned.csv')
installments_payments_cleaned = pd.read_csv(data_path+'installments_payments_cleaned.csv')
credit_card_balance_cleaned = pd.read_csv(data_path+'credit_card_balance_cleaned.csv')
bureau_balance_bureau_cleaned = pd.read_csv(data_path+'bureau_balance_bureau_cleaned.csv')
application_train_cleaned = pd.read_csv(data_path+'application_train_cleaned.csv')

# Merge all datasets according to the SK_ID_CURR column
data_merge= application_train_cleaned.merge(bureau_balance_bureau_cleaned, on='SK_ID_CURR')
data_merge = data_merge.merge(credit_card_balance_cleaned,on='SK_ID_CURR')
data_merge = data_merge.merge(installments_payments_cleaned,on='SK_ID_CURR')
data_merge = data_merge.merge(pos_cash_cleaned,on='SK_ID_CURR')
data_merge = data_merge.merge(previous_applications_cleaned,on='SK_ID_CURR')

# Drop the first two columns
data_merge = data_merge.drop(columns=['Unnamed: 0'])

# Drop the gender_code column, for ethical reasons (no discrimination considering the person's gender)
data_merge = data_merge.drop(columns=['CODE_GENDER'])

# Set the SK_ID_CURR column as index
data_merge = data_merge.set_index('SK_ID_CURR')

# Remove columns that contain more than 30% of NaN values using custom function
remove_columns_according_to_nan_percentage(data_merge,NAN_PERCENTAGE_CLEANING)

# Fill missing values with median (could be optimized with kNN according to categorical variables ?)
data_merge = data_merge.fillna(data_merge.median())

# Drop columns that contain only one distinct value
for col in data_merge.columns:
    if len(data_merge[col].unique()) == 1:
        data_merge.drop(col,inplace=True,axis=1)
        

def detect_highly_correlated_features(df:pd.DataFrame, correlation_coefficient:float=0.95)->List:
    """From a pandas dataframe containing features values as columns, detect groups of highly correlated features according to a correlation coefficient
    threshold.

    Args:
        df (pd.DataFrame): pandas dataframe containing features as columns.
        correlation_coefficient (float, optional): The correlation coefficient among which two features are considered highly correlated. Defaults to 0.95.

    Returns:
        List: List containing highly correlated features in nested lists.
    """
    
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than correlation_coefficient
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_coefficient)]

    # Get groups of highly correlated features (positively or negatively, >correlation_coefficient)
    df = df[to_drop]
    corrMatrix=df.corr()
    corrMatrix.loc[:,:] = np.tril(corrMatrix, k=-1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][abs(corrMatrix[col] > correlation_coefficient)].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
            
    return result

highly_correlated_features = detect_highly_correlated_features(data_merge, CORRELATION_COEFFICIENT)

def select_feature_from_highly_correlated_group(correlated_groups:List, keeping_method:str="mean")-> List:
    """From a list of nested lists containing groups of highly correlated features, keep one that is representative of the group.
    The selection is made according to the keeping_method, meaning that the feature containing the keyword indicated by the keeping_method
    will be kept and others will be discarded.

    Args:
        correlated_groups (List): List containing highly correlated features in nested lists.
        keeping_method (str, optional): The feature in the group will be kept if it contains this keyword, otherwise another value will be kept
        according to this order: mean>sum>var>max>min. If none of these keywords are contained in the variable name, the first one in the list will
        be selected. Defaults to "mean".

    Returns:
        List: A list containing kept features.
    """
    
    features_to_keep = []
    
    # Ordered types of feature to select, remove the wanted one from this list
    types_of_feature=["mean","sum","var","max","min"]
    types_of_feature.remove(keeping_method)
    
    for group in correlated_groups:
        
        # If there is a variable that contains the desired keyword, keep it
        keeping_feature = next((s for s in group if keeping_method in s.lower()), None)
        if keeping_feature:
            features_to_keep.append(keeping_feature)
            
        # If there is no variable that contain the desired keyword, iterate over the group to keep the feature according to the previous list
        if not keeping_feature:
            for type_of_feature in types_of_feature:
                keeping_feature = next((s for s in group if type_of_feature in s.lower()), None)
                if keeping_feature:
                    features_to_keep.append(keeping_feature)
                    break
                
        # If there is no variable that contains any of the keywords in types_of_feature list, select the first variable
        if not keeping_feature:
            features_to_keep.append(group[0])
            
    return features_to_keep

# Keep the selected features and remove the others
features_to_keep = select_feature_from_highly_correlated_group(highly_correlated_features, "mean")
flat_list_correlated_features = [item for sublist in highly_correlated_features for item in sublist]
features_to_drop = [feature for feature in flat_list_correlated_features if feature not in features_to_keep]
df_corr = data_merge.drop(columns=features_to_drop)

# Remove outliers for specified features according to values that seemed aberrant for these features
df_corr = df_corr[df_corr['CNT_CHILDREN']<7]
df_corr = df_corr[df_corr['AMT_INCOME_TOTAL']<(2*1e6)]
df_corr = df_corr[df_corr['CNT_FAM_MEMBERS']<9]                                                             
df_corr = df_corr[df_corr['OBS_30_CNT_SOCIAL_CIRCLE']<50]                                                               
df_corr = df_corr[df_corr['DEF_30_CNT_SOCIAL_CIRCLE']<10]    
df_corr = df_corr[df_corr['DEF_60_CNT_SOCIAL_CIRCLE']<10] 
df_corr = df_corr[df_corr['AMT_REQ_CREDIT_BUREAU_QRT']<20]                                                             
df_corr = df_corr[df_corr['INCOME_CREDIT_PERC']<6]                                                               
df_corr = df_corr[df_corr['INCOME_PER_PERSON']<(2*1e6)]                                                              
df_corr = df_corr[df_corr['BURO_AMT_CREDIT_SUM_DEBT_SUM']<(3*1e8)]                                                             
df_corr = df_corr[df_corr['CLOSED_AMT_CREDIT_SUM_OVERDUE_MEAN']<2000]  
df_corr = df_corr[df_corr['CC_MONTHS_BALANCE_MAX']>-6] 

# Clean the dataset to remove potential infinite or NaN values
df_corr_cleaned = clean_dataset(df_corr)

#----- Select the most important features using selectKbest with chi²

# First define training sets using SMOTE to oversample the class 1 of target values
X = df_corr_cleaned.drop(columns=['TARGET'])
y = df_corr_cleaned['TARGET']

# Normalize features between 0 and 1
scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)
scaled_X_df = pd.DataFrame(scaled_X, columns = X.columns)

X_train, X_test, y_train, y_test = train_test_split(scaled_X_df, y, test_size = 0.30)
oversample = SMOTE()
X_sm, y_sm = oversample.fit_resample(X_train, y_train)

def chi_square(X_train, y_train, n):
    selector = SelectKBest(chi2, k=n)
    selector.fit(X_train, y_train)
    cols = selector.get_support(indices=True)
    cols_names = list(X_train.iloc[:, cols].columns)
    return cols_names

# Select the 20 most important features using the chi² test
important_features = chi_square(X_sm,y_sm,20)

# Separate features and target
df_features = df_corr_cleaned[important_features]
df_target = df_corr_cleaned['TARGET']

# Save cleaned data for features and target in csv format
df_features.to_csv(data_path+"cleaned_data_most_important_features.csv")
df_target.to_csv(data_path+"cleaned_data_most_important_target.csv")
