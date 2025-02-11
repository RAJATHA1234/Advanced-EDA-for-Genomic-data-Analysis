# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.stats as stats

# Load the dataset from CSV
df = pd.read_csv('genomic_variants.csv')

df.head()

# **Basic Data Exploration**
print(df.info())  # Info about data types and non-null counts
print(df.head())  # First few rows of the data
print(df.isnull().sum())  # Check for missing values

# **Data Preprocessing**
# Handling missing values (if any)
df.dropna(subset=['Gene Name', 'Variant Effect'], inplace=True)

print(df.isnull().sum())

#Outlier Detection using Z-score:
from scipy.stats import zscore
numerical_columns = ['Variant Frequency', 'Expression Level']
df[numerical_columns] = df[numerical_columns].apply(zscore)

df_cleaned = df[(np.abs(df[numerical_columns]) < 3).all(axis=1)]

# Encoding categorical variables
df['Gene Name'] = df['Gene Name'].astype('category')
df['Variant Type'] = df['Variant Type'].astype('category')
df['Genotype'] = df['Genotype'].astype('category')
df['Phenotype'] = df['Phenotype'].astype('category')
df['Clinical Significance'] = df['Clinical Significance'].astype('category')


# **Handling Outliers**
# Detecting outliers using Z-score or IQR method
from scipy.stats import zscore

# Calculate Z-scores for numerical columns
numerical_columns = ['Variant Frequency', 'Expression Level']
df[numerical_columns] = df[numerical_columns].apply(zscore)

# Filter out rows where Z-score is greater than 3 (outliers)
df_cleaned = df[(np.abs(df[numerical_columns]) < 3).all(axis=1)]

# Alternatively, use IQR for detecting outliers:
# Q1 = df['Variant Frequency'].quantile(0.25)
# Q3 = df['Variant Frequency'].quantile(0.75)
# IQR = Q3 - Q1
# df_cleaned = df[(df['Variant Frequency'] >= (Q1 - 1.5 * IQR)) & (df['Variant Frequency'] <= (Q3 + 1.5 * IQR))]

print(f"Data shape before cleaning: {df.shape}")
print(f"Data shape after cleaning: {df_cleaned.shape}")

# Import necessary libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# **Feature Scaling (Standardization)**
# Create an instance of StandardScaler
scaler_standard = StandardScaler()

# Apply standardization (mean=0, std=1)
df_scaled = df.copy()  # Make a copy to preserve the original dataframe
df_scaled[['Variant Frequency', 'Expression Level']] = scaler_standard.fit_transform(df[['Variant Frequency', 'Expression Level']])

# **Normalization (Min-Max Scaling)**
# Create an instance of MinMaxScaler (scales between 0 and 1)
scaler_minmax = MinMaxScaler()

# Apply normalization (scales between 0 and 1)
df_normalized = df.copy()  # Make a copy to preserve the original dataframe
df_normalized[['Variant Frequency', 'Expression Level']] = scaler_minmax.fit_transform(df[['Variant Frequency', 'Expression Level']])

# Display the scaled and normalized data
print("\nScaled Data (Standardization - Mean=0, Std=1):")
print(df_scaled[['Variant Frequency', 'Expression Level']].head())

print("\nNormalized Data (Min-Max Scaling - Range [0, 1]):")
print(df_normalized[['Variant Frequency', 'Expression Level']].head())

# Optionally, you can visualize the difference:
# Plot to compare the original, scaled, and normalized data distributions
plt.figure(figsize=(12, 6))

# Original data distribution
plt.subplot(1, 3, 1)
sns.histplot(df[['Variant Frequency', 'Expression Level']], bins=30, kde=True)
plt.title('Original Data Distribution')

# Scaled data distribution (Standardized)
plt.subplot(1, 3, 2)
sns.histplot(df_scaled[['Variant Frequency', 'Expression Level']], bins=30, kde=True)
plt.title('Standardized Data (Mean=0, Std=1)')

# Normalized data distribution
plt.subplot(1, 3, 3)
sns.histplot(df_normalized[['Variant Frequency', 'Expression Level']], bins=30, kde=True)
plt.title('Normalized Data (Range [0, 1])')

plt.tight_layout()
plt.show()

# **Visualization - Exploring Variant Frequency**
plt.figure(figsize=(10, 6))
sns.histplot(df['Variant Frequency'], bins=30, kde=True)
plt.title('Distribution of Variant Frequency')
plt.xlabel('Variant Frequency')
plt.ylabel('Count')
plt.show()

gene_genotype_matrix = pd.crosstab(df['Gene Name'], df['Genotype'])

# Compute the co-occurrence matrix (how often each gene/genotype pair appears together)
co_occurrence_matrix = gene_genotype_matrix.dot(gene_genotype_matrix.T)

# Plot the co-occurrence heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(co_occurrence_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap of Variant Co-occurrence (Genes vs Genotypes)')
plt.xlabel('Gene')
plt.ylabel('Gene')
plt.show()

# **Pairwise Plots (Scatter Matrix) for Multiple Numerical Variables**
# A pairplot will help explore relationships between multiple variables like 'Variant Frequency' and 'Expression Level'

# Select numerical columns for pairwise plot
numerical_columns = ['Variant Frequency', 'Expression Level']

# For more comprehensive analysis, we can also include additional numerical features if they exist in your dataset
sns.pairplot(df[numerical_columns])
plt.suptitle('Pairwise Plot (Scatter Matrix)', y=1.02)
plt.show()

# If you want to include more variables in the pairplot (for example, add a categorical variable like 'Phenotype' for color-coding)
sns.pairplot(df[numerical_columns + ['Phenotype']], hue='Phenotype')
plt.suptitle('Pairwise Plot (Scatter Matrix) with Phenotype', y=1.02)
plt.show()

# Use pairplot with additional customization to include correlation
sns.pairplot(df[numerical_columns], kind='scatter', plot_kws={'alpha': 0.6})
plt.suptitle('Advanced Pairwise Plot with Correlation', y=1.02)
plt.show()

# **Variant Effect by Phenotype**
plt.figure(figsize=(12, 6))
sns.countplot(x='Variant Effect', hue='Phenotype', data=df)
plt.title('Variant Effect by Phenotype')
plt.xlabel('Variant Effect')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# **Variant Frequency vs Expression Level**
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Variant Frequency', y='Expression Level', data=df, hue='Phenotype')
plt.title('Variant Frequency vs Expression Level')
plt.xlabel('Variant Frequency')
plt.ylabel('Expression Level')
plt.show()

# **PCA (Principal Component Analysis) - Dimensionality Reduction**
X = df[['Variant Frequency', 'Expression Level']]  # Features for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue=df['Phenotype'])
plt.title('PCA of Genomic Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# **Clustering with KMeans**
# Using KMeans clustering to group samples into clusters based on the features
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Variant Frequency', y='Expression Level', hue='Cluster', data=df, palette='viridis')
plt.title('KMeans Clustering of Genomic Data')
plt.xlabel('Variant Frequency')
plt.ylabel('Expression Level')
plt.show()


#optional
from sklearn.cluster import KMeans

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Variant Frequency', y='Expression Level', hue='Cluster', data=df, palette='viridis')
plt.title('Clustering of Samples Based on Genetic Features')
plt.xlabel('Variant Frequency')
plt.ylabel('Expression Level')
plt.show()
# **Random Forest Classifier to Predict Phenotype**
# Encoding Phenotype as binary values
df['Phenotype_binary'] = df['Phenotype'].map({'Affected': 1, 'Unaffected': 0})

# Features and target
X_ml = df[['Variant Frequency', 'Expression Level']]
y_ml = df['Phenotype_binary']
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import models

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Assuming 'df' is your dataframe and already loaded
# Encoding Phenotype as binary values
df['Phenotype_binary'] = df['Phenotype'].map({'Affected': 1, 'Unaffected': 0})

# Features and target
X_ml = df[['Variant Frequency', 'Expression Level']]
y_ml = df['Phenotype_binary']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)

# **Feature Scaling**
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# **Autoencoder Model Design**
input_layer = Input(shape=(X_train_scaled.shape[1],))  # Input layer (number of features)

# Encoder: compress the data to a smaller dimension
encoded = Dense(2, activation='relu')(input_layer)  # Example: 2 nodes in the encoded layer

# Decoder: reconstruct the data from the compressed representation
decoded = Dense(X_train_scaled.shape[1], activation='sigmoid')(encoded)

# Build the autoencoder model
autoencoder = Model(input_layer, decoded)

# **Compile the model**
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# **Train the Autoencoder**
autoencoder.fit(X_train_scaled, X_train_scaled, epochs=50, batch_size=256, validation_data=(X_test_scaled, X_test_scaled))

# **Extract Encoded Features**
encoder = Model(input_layer, encoded)  # We only need the encoder part to get the compressed representation

X_train_encoded = encoder.predict(X_train_scaled)  # Encode the training data
X_test_encoded = encoder.predict(X_test_scaled)  # Encode the test data

# **Train a Random Forest Classifier on Encoded Features**
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_encoded, y_train)

# **Making Predictions**
y_pred = rf_clf.predict(X_test_encoded)

# **Evaluate Model**
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_rep}")

# Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# **Feature Importance**
plt.figure(figsize=(8, 6))
sns.barplot(x=X_ml.columns, y=rf_clf.feature_importances_)
plt.title('Feature Importance for Phenotype Prediction')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

# **T-test to Compare Variant Frequency Between Affected and Unaffected**
affected = df[df['Phenotype'] == 'Affected']['Variant Frequency']
unaffected = df[df['Phenotype'] == 'Unaffected']['Variant Frequency']
t_stat, p_value = stats.ttest_ind(affected, unaffected)

print(f"T-statistic: {t_stat}, P-value: {p_value}")



