import streamlit as st
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="GMM - Shopping Patterns", layout="centered")
st.title("🛍️ Day 20 — Gaussian Mixture Model: Cluster Shopping Patterns")

@st.cache_data
def gen_customers(n=600, seed=42):
    rng = np.random.RandomState(seed)
    # features: frequency, avg_spend, recency (days since last)
    c1 = np.column_stack([rng.normal(50,10, n//3), rng.normal(20,5, n//3), rng.normal(10,4, n//3)])  # frequent small spenders
    c2 = np.column_stack([rng.normal(10,4, n//3), rng.normal(150,40, n//3), rng.normal(60,20, n//3)]) # infrequent big spenders
    c3 = np.column_stack([rng.normal(30,8, n//3), rng.normal(60,20, n//3), rng.normal(30,8, n//3)])   # medium
    X = np.vstack([c1, c2, c3])
    df = pd.DataFrame(X, columns=["frequency", "avg_spend", "recency"])
    return df

df = gen_customers()
st.subheader("📂 Sample Customers")
st.dataframe(df.sample(8))

# scale
scaler = StandardScaler()
Xs = scaler.fit_transform(df)

# controls
n_components = st.slider("Number of GMM components", 2, 6, 3)

gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(Xs)
probs = gmm.predict_proba(Xs)
labels = gmm.predict(Xs)
df["segment"] = labels
df["max_prob"] = probs.max(axis=1)

st.write(f"Found {n_components} soft clusters. Here are counts:")
st.write(df["segment"].value_counts().sort_index())

# Plot pairplot colored by hard label
fig = plt.figure(figsize=(8,6))
sns.scatterplot(x=df["avg_spend"], y=df["frequency"], hue=df["segment"], palette="tab10")
plt.xlabel("Average Spend")
plt.ylabel("Frequency")
plt.title("Shopping Segments (GMM)")
st.pyplot(fig)

# Show example of soft membership for first customers
st.subheader("🔎 Example soft-membership probabilities (first 6 customers)")
st.dataframe(pd.DataFrame(probs[:6], columns=[f"comp_{i}" for i in range(probs.shape[1])]))
st.success("✅ GMM clustering (soft assignments) completed")
