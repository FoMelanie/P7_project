import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Import metrics values
model_results_custom_scorer = pd.read_csv('metrics_comparison_custom_scorer.csv')

# Rework the dataframe
model_results_custom_scorer = model_results_custom_scorer.drop(columns=['Unnamed: 0'])
model_results_models = model_results_custom_scorer[
    [
        "model",
        "best_param",
        "cv_score",
        "test_score",
        "mean_train_accuracy",
        "mean_train_precision",
        "mean_train_recall",
        "mean_train_f1_score",
        "mean_val_accuracy",
        "mean_val_precision",
        "mean_val_recall",
        "mean_val_f1_score",
        "classifier",
    ]
]

# Add dummy data 
model_results_models.iloc[0, 0] = 'dummy'
model_results_models['mean_train_accuracy'][model_results_models['model'] == 'dummy'] = model_results_custom_scorer['cv_model_mean_train_accuracy'][0]
model_results_models['mean_train_precision'][model_results_models['model'] == 'dummy'] = model_results_custom_scorer['cv_model_mean_train_precision'][0]
model_results_models['mean_train_recall'][model_results_models['model'] == 'dummy'] = model_results_custom_scorer['cv_model_mean_train_recall'][0]
model_results_models['mean_train_f1_score'][model_results_models['model'] == 'dummy'] = model_results_custom_scorer['cv_model_mean_train_f1_score'][0]
model_results_models['mean_val_accuracy'][model_results_models['model'] == 'dummy'] = model_results_custom_scorer['cv_model_mean_val_accuracy'][0]
model_results_models['mean_val_precision'][model_results_models['model'] == 'dummy'] = model_results_custom_scorer['cv_model_mean_val_precision'][0]
model_results_models['mean_val_recall'][model_results_models['model'] == 'dummy'] = model_results_custom_scorer['cv_model_mean_val_recall'][0]
model_results_models['mean_val_f1_score'][model_results_models['model'] == 'dummy'] = model_results_custom_scorer['cv_model_mean_val_f1_score'][0]
model_results_models['classifier'][model_results_models['model'] == 'dummy'] = 'dummy classifier'

# Plot metric values and export the figures
metric_to_plot = [
    "mean_train_accuracy",
    "mean_train_precision",
    "mean_train_recall",
    "mean_train_f1_score",
    "mean_val_accuracy",
    "mean_val_precision",
    "mean_val_recall",
    "mean_val_f1_score",
]

for metric in metric_to_plot:
    fig = plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=model_results_models, x="model", y=metric)
    plt.xticks(rotation=90)
    plt.savefig(f"{metric}.png")