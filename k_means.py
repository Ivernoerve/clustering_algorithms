import numpy as np
import matplotlib.pyplot as plt



class K_means():
    
    def __init__(self, data: np.ndarray, n_clusters: int) -> None:
        """
        K_means clustering algorithm.
        data: data to apply the clustering to in format samples, features.
        n_clusters: number of clusters to divide into
        """
        self.data = data
        self.n_clusters = n_clusters
        self._initiate_clusters()
        
    def _initiate_clusters(self):
        """
        Method to initiate the clusters for the K means
        selecting a random point from the data to become the label
        """
        cluster_indexes = np.random.randint(self.data.shape[0], size = self.n_clusters)
    
        self.cluster_array = self.data[cluster_indexes, :]
        self.cluster_distance = np.empty((self.n_clusters, self.data.shape[0]), dtype = object)
        self.old_cluster_data_index = np.ones(self.data.shape[0]) * self.n_clusters
        self.cluster_data_indexes = np.zeros(self.data.shape[0])
        return self


    def _calculate_distances(self):
        """
        Method to clalulate the euclidian distance to every point
        creating a distance matrix containing the distance from every cluster to the data
        """
        for i, cluster in enumerate(self.cluster_array):

            distance = np.linalg.norm(self.data - cluster, axis = 1)

            self.cluster_distance[i] = distance

    def _assign_to_cluster(self):
        """
        Method to assign data points to the clusters based on distance
        """
        self.cluster_data_indexes = np.argmin(self.cluster_distance, axis = 0)

        for i in range(self.n_clusters):
            cluster_index = np.where(self.cluster_data_indexes == i)[0]
            if len(cluster_index) != 0:
                self.cluster_array[i] = np.mean(self.data[cluster_index], axis = 0)

        return self


    def assign_classes(self):
        """
        Method to assign labels to the data. 
        Runs until convergence
        """
        self.states = []
        while np.array_equal(self.old_cluster_data_index, self.cluster_data_indexes) != True:
            
            self.old_cluster_data_index = self.cluster_data_indexes
            self._calculate_distances()
            self._assign_to_cluster()
            self.states.append(self.cluster_data_indexes)


if __name__ == "__main__":

    cov = np.array([[1,0],[0,1]])
    cov2 =np.array([[5,0],[0,5]])


    

    class_0_1 = np.random.multivariate_normal([5,10], cov, (100))
    class_0_2 = np.random.multivariate_normal([4,4], cov, (100))
    class_1 = np.random.multivariate_normal([15, 10], cov , (100))
    class_2 = np.random.multivariate_normal([10, 10], cov2 , (100))
    class_3 = np.random.multivariate_normal([15, 4], cov2 , (100))


    data2d = np.vstack((class_0_1, class_0_2, class_1, class_2, class_3))

    cov3d = np.array([[1 ,0, 0],[0, 10, 0], [0, 0, 1]])



    

    class_0_1 = np.random.multivariate_normal([5,10, 2], cov3d, (100))
    class_0_2 = np.random.multivariate_normal([4,4, 2], cov3d, (100))
    class_1 = np.random.multivariate_normal([15, 10, 6], cov3d , (100))
    class_2 = np.random.multivariate_normal([10, 10, 0], cov3d , (100))
    class_3 = np.random.multivariate_normal([15, 4, 10], cov3d , (100))


    data3d = np.vstack((class_0_1, class_0_2, class_1, class_2, class_3))


    a = K_means(data2d, 4)
    a._calculate_distances()
    a._assign_to_cluster()
    a.assign_classes()


    def plot_2d(k_means2d):
        for state in k_means2d.states:
            plt.figure()
            for i in range(k_means2d.n_clusters):
                cluster_index = np.where(state == i)[0]

                plt.scatter(k_means2d.data[cluster_index, 0], k_means2d.data[cluster_index, 1], label = f"{i}")
        
            plt.legend()
            plt.pause(3)
            plt.close()

        


        return 0


    def plot_3d(k_means3d):
        for state in k_means3d.states:
            plt.figure()
            ax = plt.axes(projection ="3d")
            
            for i in range(k_means3d.n_clusters):
                cluster_index = np.where(state == i)[0]

                ax.scatter3D(k_means3d.data[cluster_index, 0], k_means3d.data[cluster_index, 1], k_means3d.data[cluster_index, 2])

        
            plt.legend()
            plt.pause(0.5)
            plt.close()


       
        return 0


    """
    a = K_means(data3d, 6)

    a._calculate_distances()
    a._assign_to_cluster()
    a.assign_classes()
    """
    plot_2d(a)
