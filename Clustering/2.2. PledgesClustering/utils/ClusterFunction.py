""" Importing Relevant Libraries """
from sklearn.cluster import KMeans # For applying K-Means clustering

# Function to find optimal cluster: x = the data to cluster , c = the number of clusters to find
def OptiCluster(x, c):
    """
    OptiCluster find the optimal clusters for a given c based on the inertia criterion

    :param x: The data to cluster (np array)
    :param c: Number of clusters to find (int)
    :return: Array containing the predicted clusters
    """

# Initialisation of a first k-means algorithm
    kmeans = KMeans(n_clusters=c, random_state=0)
    kmeans.fit(x)

    inertia = kmeans.inertia_ # Quality criterion --> The lower the inertia the better the clusters
    print(inertia)

    yKm = kmeans.fit_predict(x) # Storing the predicted clusters in a temporary variable

# Repeating the process 500 times to find an optimum
    for i in range(500): 
        kmeansN = KMeans(n_clusters=c, random_state=i+1)
        kmeansN.fit(x)
# Change the values of yKm and inertia only if the new clusters have a better quality
        if kmeansN.inertia_ < inertia: 
            inertia = kmeansN.inertia_
            yKm = kmeansN.fit_predict(x)
    
    print(inertia)

    return yKm