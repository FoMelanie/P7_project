import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import re
from sklearn.model_selection import train_test_split
import seaborn as sns
import shap
import lime
import dill

#--- Import needed data

# Import XGBoost model with custom scorer function and scale_pos_weight
xgb_model = joblib.load('3_models/xgb_weight_pipeline_custom_scorer.joblib')

# Import cleaned data (features and target) as pandas dataframes
features = pd.read_csv('1_feature_engineering_cleaning/cleaned_data_most_important_features.csv')
target = pd.read_csv('1_feature_engineering_cleaning/cleaned_data_target.csv')

# Merge the two dataframes and select a sample with SK_ID_CURR as index (10% of total data)
df_full = pd.merge(features, target, how ='inner', on =['SK_ID_CURR'])
sample_df = df_full.sample(frac=0.1, replace=True, random_state=1)
sample_df = sample_df.set_index('SK_ID_CURR')

# Rename columns to avoid special json characters
sample_df = sample_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# Select X and y
X = sample_df.drop('TARGET', axis=1)
y = sample_df['TARGET']

# Export X
X.to_csv('5_best_model_explicability/features_shap_values.csv')

# Define train and test sets from original features data that will be used for explicability
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3)

# Fit model to this training test
xgb_model.fit(X_train, Y_train)

# Determine predicted values from loaded model and train/test sets
y_pred = xgb_model.predict(X_test)

#--- General feature importance

color_list =  sns.color_palette("dark", len(X.columns)) 
top_x = 10 # 10 most important features to show

fig, axs = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')

feature_importance = xgb_model[-1].feature_importances_
indices = np.argsort(feature_importance)
indices = indices[-top_x:]

bars = axs.barh(range(len(indices)), feature_importance[indices], color='b', align='center') 
axs.set_title("XGBoost", fontweight="normal", fontsize=16)
axs.tick_params(axis='both', which='major', labelsize=15)

plt.sca(axs)
plt.yticks(range(len(indices)), [X.columns[j] for j in indices], fontweight="normal", fontsize=15) 

for i, ticklabel in enumerate(plt.gca().get_yticklabels()):
    ticklabel.set_color(color_list[indices[i]])  
for i,bar in enumerate(bars):
    bar.set_color(color_list[indices[i]])
    plt.box(False)

plt.suptitle("Top " + str(top_x) + " Features.", fontsize=12, fontweight="normal")
plt.savefig('5_best_model_explicability/general_feature_importance.png', bbox_inches='tight')

#--- Explicability with lime (local)

# Create an explainer instance
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(), 
                                                   feature_names=X.columns, 
                                                   class_names=['safe','risk'], 
                                                   verbose=True, 
                                                   mode='classification')

# Export the explainer instance with dill for API further usage
with open("5_best_model_explicability/lime_explainer_xgb", 'wb') as f: dill.dump(explainer, f)

#--- Explicability with shap values (global and local)

# Generate shap values and export them
shap_explainer = shap.TreeExplainer(xgb_model[-1])
shap_values = shap_explainer(X)
with open('5_best_model_explicability/shap_values_xgb', 'wb') as f: dill.dump(shap_values, f)
with open('5_best_model_explicability/shap_explainer_xgb', 'wb') as f: dill.dump(shap_explainer, f)

# General explicability
fig, axs = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
axs = shap.summary_plot(shap_values, X, plot_type="bar")
plt.savefig('5_best_model_explicability/shap_general_feature_importance.png', bbox_inches='tight')