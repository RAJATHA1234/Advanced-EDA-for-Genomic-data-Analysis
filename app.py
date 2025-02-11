import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set page title and layout
st.set_page_config(page_title="Genomic Data Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('genomic_variants.csv')
    df.dropna(subset=['Gene Name', 'Variant Effect'], inplace=True)
    df['Gene Name'] = df['Gene Name'].astype('category')
    df['Variant Type'] = df['Variant Type'].astype('category')
    df['Genotype'] = df['Genotype'].astype('category')
    df['Phenotype'] = df['Phenotype'].astype('category')
    return df

# Sidebar menu
st.sidebar.header("Genomic Data Analysis")
menu_options = ["Explore Data", "PCA Analysis", "Phenotype Prediction"]
selection = st.sidebar.radio("Select an option", menu_options)

# Display home page
if selection == "Explore Data":
    st.title("Explore Genomic Data")
    df = load_data()

    st.write("Data Overview:")
    st.dataframe(df.head())

    st.write("Variant Frequency Distribution:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Variant Frequency'], bins=30, kde=True, ax=ax)
    ax.set_title('Distribution of Variant Frequency')
    ax.set_xlabel('Variant Frequency')
    ax.set_ylabel('Count')
    st.pyplot(fig)

elif selection == "PCA Analysis":
    st.title("Principal Component Analysis (PCA)")
    df = load_data()

    # Select the features for PCA
    X = df[['Variant Frequency', 'Expression Level']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])

    # Show PCA scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue=df['Phenotype'], ax=ax)
    ax.set_title('PCA of Genomic Data')
    st.pyplot(fig)

elif selection == "Phenotype Prediction":
    st.title("Predict Phenotype Based on Genomic Data")

    # Input features
    variant_frequency = st.number_input("Variant Frequency", min_value=0.0, max_value=1.0, value=0.5)
    expression_level = st.number_input("Expression Level", min_value=0.0, max_value=10.0, value=2.0)

    # Prepare data for prediction
    df = load_data()
    X = df[['Variant Frequency', 'Expression Level']]
    y = df['Phenotype'].map({'Affected': 1, 'Unaffected': 0})

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make prediction on user input
    if st.button("Predict"):
        prediction = model.predict([[variant_frequency, expression_level]])
        prediction_result = "Affected" if prediction[0] == 1 else "Unaffected"
        st.write(f"Predicted Phenotype: {prediction_result}")
