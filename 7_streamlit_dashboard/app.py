import streamlit as st
import pandas as pd
import requests
import json
import shap
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import lime
import shap
import dill
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dashboard_functions import cluster_comparison_bar
st.set_option('deprecation.showPyplotGlobalUse', False)

#--- Import needed data

# Import cleaned dataframe with customer id (first 100) and features values
df = pd.read_csv("cleaned_data_most_important_features.csv")
df_sample = df.head(100)
features_shap_values = pd.read_csv("features_shap_values.csv").set_index('SK_ID_CURR')

# Set SK_ID_CURR as index for explainability
df_indexed = df_sample.set_index('SK_ID_CURR')
df_indexed_full = df.set_index('SK_ID_CURR')

# Import saved model for interpretability analysis
model = joblib.load('xgb_weight_pipeline_custom_scorer.joblib')

# Import explainers (lime and shap)
with open('lime_explainer_xgb', 'rb') as lime_explainer: lime_explainer = dill.load(lime_explainer)
with open('shap_values_xgb', 'rb') as shap_values: shap_values = dill.load(shap_values)
with open('shap_explainer_xgb', 'rb') as shap_explainer: shap_explainer = dill.load(shap_explainer)

#--- Run dashboard

# Enter customer ID to predict if a home credit would be predicted or not
def run():
    
    st.title("Home Credit Prediction")
    st.markdown("""Welcome to \"Prêt à dépenser\" home credit prediction. Please select a customer ID on the left panel to see its personal information in the table below.
                A customer classification model allows to determine the risk level of according a loan to the selected customer. The associated prediction is indicated below.
                Transparency is important to us: further information regarding customers characteristics are available to decipher what impacts the previous prediction.""")
    st.sidebar.image("logo_P7.png", use_column_width=True)
    st.sidebar.header("Credit prediction")
    st.sidebar.markdown("Please select a customer ID from the list below")
    customer_list = df['SK_ID_CURR'].tolist()
    customer_id = st.sidebar.selectbox('Select a customer ID', customer_list)
    
    # Get customer data and print it for the 20 most important features
    st.markdown(f'<h1 style="color:#3598B2;font-size:24px;">{"Customer personal informations"}</h1>', unsafe_allow_html=True)
    data = df.loc[df["SK_ID_CURR"] == customer_id]
    data = data.set_index("SK_ID_CURR")
    st.dataframe(data=data)
    st.markdown("""---""")
    
    # Predict using the model by requesting fastapi. Convert the response to dict to access probability
    response = requests.get(url = "https://fastapi-backend-p7.herokuapp.com/predict/" + str(customer_id))
    response = response.text
    response=json.loads(response)
    
    # Interpret results: if the probability of belonging to class "1" (not repay) is low, the customer is predicted as "low risk"
    st.markdown(f'<h1 style="color:#3598B2;font-size:24px;">{"Customer prediction"}</h1>', unsafe_allow_html=True)
    if response["Prediction"]<0.5:
        customer_risk = "low risk customer"
        st.success(f"This customer is a **{customer_risk}**. Its calculated probability of not repaying its loan is {round(response['Prediction'], 2)}.")
        
    else:
        customer_risk= "higher risk customer"
        st.error(f"This customer is a **{customer_risk}**. Its calculated probability of not repaying its loan is {round(response['Prediction'], 2)}.")
    st.markdown("""---""")
    
    # Model interpretability
    st.markdown(f'<h1 style="color:#3598B2;font-size:24px;">{"Customer interpretability"}</h1>', unsafe_allow_html=True)

    # General feature importance
    st.markdown(f'<h2 style="color:#3598B2;font-size:24px;">{"General feature importance"}</h2>', unsafe_allow_html=True)
    st.markdown("The top 10 features that are the most important in classifying the customers are presented in the figure below, ordered by importance.")
    st.image('general_feature_importance.png')
    
    # Summary scatterplot
    st.markdown("Below is a summary plot showing the impact of the selected features on the classification. For example, The higher the \"highereducation\" values are, the more it impacts negatively the classification, pushing towards class 0. Then, we can deduce that a low education level is one of the factor explaining the absence or delay of loan payment.")
    st.pyplot(shap.summary_plot(shap_values, features_shap_values))
    
    # Local feature importance : at the customer scale with LIME
    # Select a particular instance for Explanations: according to customer SK_ID_CURR
    st.markdown(f'<h2 style="color:#3598B2;font-size:24px;">{"Customer explicability"}</h2>', unsafe_allow_html=True)
    test_array = df_indexed.loc[[customer_id]].to_numpy()
    reshaped_test_array = test_array.ravel()

    lime_exp = lime_explainer.explain_instance(
                                    reshaped_test_array, 
                                    model.predict_proba, 
                                    num_features=5
                                    )   
    lime_exp.save_to_file('lime.html')
    
    # Read file and keep in variable
    with open('lime.html','r', encoding='utf-8') as f: 
        html_data = f.read()
    st.components.v1.html(html_data,height=200)
    
    st.markdown("""---""")
    
# Example of target = 1 : 100047

    # Similar customers compared to the selected one and their characteristics
    st.markdown(f'<h1 style="color:#3598B2;font-size:24px;">{"Similar customers"}</h1>', unsafe_allow_html=True)

    # Initialize kmeans parameters
    kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "random_state": 1,
    }
    
    #create scaled DataFrame where each variable has mean of 0 and standard dev of 1
    scaled_df = StandardScaler().fit_transform(df_indexed)
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_df)
        sse.append(kmeans.inertia_)
    
    #instantiate the k-means class, using optimal number of clusters
    kmeans = KMeans(init="random", n_clusters=5, n_init=10, random_state=1)

    #fit k-means algorithm to data
    kmeans.fit(scaled_df)
    
    #append cluster assingments to original DataFrame
    df_indexed['cluster'] = kmeans.labels_
    customer_cluster = df_indexed.loc[customer_id,'cluster']
    st.markdown(f"The customer {customer_id} belongs to customer cluster {customer_cluster}.")
    
    # State the cluster correspondance
    st.markdown("Please select the feature for which you want to see cluster distribution.")
    features_list = df_indexed.columns.values.tolist()
    features_list.remove('cluster')
    selected_feature = st.selectbox('Select a feature', features_list)
    cluster_comparison_bar(number_of_clusters=5,X_comparison=df_indexed,cluster_column='cluster',selected_feature=selected_feature)
    
    # Violinplots
    selected_feature_left = st.selectbox('Select a feature to observe customer value compared to others', features_list)
    
    # Violinplot : comparison with customers from the same cluster
    st.markdown("Comparison with customer from the same cluster (similar customers). The red line corresponds to the selected customer value.")
    fig = plt.figure(figsize=(8, 5), dpi=200)
    graph = sns.violinplot(data=df_indexed[df_indexed['cluster']==customer_cluster], x=selected_feature_left)
    graph.axvline(df_indexed.loc[customer_id,selected_feature_left],linewidth=2, color='r')
    st.pyplot(fig)
    
    # Violinplot : comparison with all customers
    st.markdown("Comparison with all customers. The red line corresponds to the selected customer value.")
    fig = plt.figure(figsize=(8, 5), dpi=200)
    graph = sns.violinplot(data=df_indexed, x=selected_feature_left)
    graph.axvline(df_indexed.loc[customer_id,selected_feature_left],linewidth=2, color='r')
    st.pyplot(fig)
# Run
if __name__ == '__main__':
    run()
