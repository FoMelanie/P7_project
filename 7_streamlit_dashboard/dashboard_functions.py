import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as st

def cluster_comparison_bar(
    number_of_clusters: int,
    X_comparison,
    selected_feature,
    cluster_column,
    deviation=True,
    title="Cluster results",
):

    palette = [
        "blue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "pink",
        "grey",
        "yellowgreen",
        "darkcyan",
        "firebrick",
        "sandybrown",
    ]
    
    # set figure size
    fig = plt.figure(figsize=(8, 5), dpi=200)

    ax = sns.barplot(
        y=X_comparison[selected_feature],
        x=X_comparison[cluster_column],
        palette=palette[0 : len(X_comparison[cluster_column].unique())],
    )
    ax.set(title=selected_feature)
    x_axis = ax.axes.get_xaxis()
    x_axis.set_visible(True)
    
    plt.show()
    st.pyplot(fig)