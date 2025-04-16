import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load your data and clean it
data = pd.read_csv("C:/Users/dk103/Documents/Market_Basket_Optimisation DA .csv")
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
data = data.replace(';', '', regex=True)

# Function for First Choices Bar Plot
def visualize_first_choices(data):
    transaction = data.values[:, 0]  # Assuming the data has only one column
    transaction = np.array(transaction)

    df_first = pd.DataFrame(transaction, columns=["items"])
    df_first["incident_count"] = 1
    df_table_first = df_first.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()

    st.subheader("First Choices Analysis")
    st.bar_chart(df_table_first.set_index("items"))

# Function for Second Choices Network Graph
def visualize_second_choices(data):
    transaction = data.values[:, 0]  # Assuming the data has only one column
    transaction = np.array(transaction)

    df_second = pd.DataFrame(transaction, columns=["items"])
    df_second["incident_count"] = 1
    indexNames = df_second[df_second['items'] == "nan"].index
    df_second.drop(indexNames, inplace=True)
    df_table_second = df_second.groupby("items").sum().sort_values("incident_count", ascending=False).reset_index()
    df_table_second["food"] = "food"
    df_table_second = df_table_second.truncate(before=-1, after=15)

    second_choice = nx.from_pandas_edgelist(df_table_second, source='food', target="items", edge_attr=True)
    pos = nx.spring_layout(second_choice)
    fig, ax = plt.subplots(figsize=(20, 20))
    nx.draw_networkx_nodes(second_choice, pos, node_size=12500, node_color="honeydew", ax=ax)
    nx.draw_networkx_edges(second_choice, pos, width=3, alpha=0.6, edge_color='black', ax=ax)
    nx.draw_networkx_labels(second_choice, pos, font_size=18, font_family='sans-serif', ax=ax)
    plt.axis('off')
    plt.grid(False)
    plt.title('Top 15 Second Choices', fontsize=25)
    st.pyplot(fig)

# Function for Apriori Analysis
def apriori_analysis(data):
    # Preprocess data for Apriori
    dataset = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    dataset = dataset.replace(';', '', regex=True)

    # Convert the data to the format required by the mlxtend library
    transactions = []
    for i in range(len(dataset)):
        transactions.append([str(dataset.values[i, j]) for j in range(len(dataset.columns))])

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Run Apriori algorithm
    frequent_itemsets = apriori(df, min_support=0.005, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

    # Display the first 25 items with highest support
    st.subheader("Apriori Analysis")
    first25 = df.columns[:25].values  # Use the first 25 columns for Apriori Analysis
    st.write("First 25 items with highest support:")
    st.write(first25)

    # Display the association rules
    st.write("Association Rules:")
    st.write(rules)

# Main function to run the Streamlit app
def main():
    st.title("Market Basket Analysis")
    
    # Sidebar with options
    choices = ["First Choices Bar Plot", "Second Choices Network Graph", "Apriori Analysis"]
    selected_choice = st.sidebar.selectbox("Select Analysis", choices)

    if selected_choice == "First Choices Bar Plot":
        visualize_first_choices(data)
    elif selected_choice == "Second Choices Network Graph":
        visualize_second_choices(data)
    elif selected_choice == "Apriori Analysis":
        apriori_analysis(data)

# Run the app
if __name__ == "__main__":
    main()
