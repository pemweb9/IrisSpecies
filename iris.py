import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv("Iris.csv")

df.rename(
    index=str,
    columns={
        "SepalLengthCm": "Sepal Length",
        "SepalWidthCm": "Sepal Width",
        "PetalLengthCm": "Petal Length",
        "PetalWidthCm": "Petal Width",
    },
    inplace=True,
)

df.drop("Id", inplace=True, axis=1)

st.header("DATASET")
st.write(df)

X = df.iloc[:, [0, 1, 2, 3]].values

wcss = []

for i in range(1, 10):
    kmeans = KMeans(
        n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 10), wcss)
plt.title("The elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()

st.set_option("deprecation.showPyplotGlobalUse", False)
elbo_plot = st.pyplot()

st.sidebar.subheader("Nilai jumlah")
cluster = st.sidebar.slider("Plih jumlah cluster :", 2, 10, 3, 1)


def k_means(best_k):
    kmeans = KMeans(random_state=0, n_clusters=best_k)
    y_kmeans = kmeans.fit_predict(X)

    plt.scatter(
        X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=50, c="#DB4CB2", label="Iris-setosa"
    )
    plt.scatter(
        X[y_kmeans == 1, 0],
        X[y_kmeans == 1, 1],
        s=50,
        c="#9966FF",
        label="Iris-versicolour",
    )
    plt.scatter(
        X[y_kmeans == 2, 0],
        X[y_kmeans == 2, 1],
        s=50,
        c="#7D3AC1",
        label="Iris-virginica",
    )

    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=50,
        c="#111111",
        label="Centroids",
    )

    plt.title(
        "There are 3 clusters formed based on the K-means implementation results.",
        fontsize=8,
        fontfamily="sans-serif",
        loc="left",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1.025])
    plt.xlabel("SepalLengthCm", fontsize=9, fontfamily="sans-serif")
    plt.ylabel("SepalWidthCm", fontsize=9, fontfamily="sans-serif")
    plt.grid(axis="both", alpha=0.2)
    plt.legend(
        title="$\\bf{Iris}$",
        fontsize=8,
        title_fontsize=9,
        loc="upper right",
        frameon=True,
    )

    st.header("Cluster Plot")
    st.pyplot()
    st.write(X)


k_means(cluster)
