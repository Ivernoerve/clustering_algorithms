import numpy as np 
import matplotlib.pyplot as plt


class Spectral_clustering():
    def __init__(self, data: np.ndarray, n_clusters: int) -> None:
        '''
        Spectral clustering implementation, to cluster data into classes. 

        :param data: The data to perform clustering with shape: samples, features.
        :param n_clusters: The number of clusters to separate the data into 
        '''
        pass

    


def generate_cluster(samples: np.ndarray, mean: np.ndarray, variance: int = 1) -> np.ndarray:
    '''
    Creates cluster with symmetric variance accros all axes 

    :param samples: The number of samples in the cluster
    :param mean: The mean of the cluster
    :param variance: The variance of the cluster (symmetrical) 

    :returns: Returns array of samples
    '''
    return np.random.multivariate_normal(mean, np.identity(mean.shape[0]) * variance, samples)


def kernel_similarity(data: np.ndarray, kernel: callable = 'gaussian') -> np.ndarray:
    '''
    Function to find the similarity meassure with a kernel method. 

    :param data: The data to find the similarities between shape, samples, features
    :param kernel: The kernel to use for similarity meassure, 
        defaults to gaussian kernel. 
    '''
    gaussian = lambda given_sample, sigma=1: np.exp(-np.linalg.norm((data - given_sample[np.newaxis]), axis = 1) ** 2 / (2 * sigma ** 2))

    
    print(data.shape)

    similarity_matrix = np.array(list(map(gaussian, data)))


    print(similarity_matrix)

if __name__ == '__main__':

    test_data = generate_cluster(10, np.array([1,1]), 2)
    
    
    kernel_similarity(test_data)

    for i, d in enumerate(test_data):
        plt.scatter(d[0], d[1], label = i)
        plt.legend() 
    plt.show()