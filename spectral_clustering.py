import numpy as np 
import matplotlib.pyplot as plt

from k_means import K_means

class Spectral_clustering():
    def __init__(self, similarity_matrix: np.ndarray, n_clusters: int) -> None:
        '''
        Spectral clustering implementation, to cluster data into classes. 

        :param similarity_matrix: Similarity matrix for the data to perform clustering on. 
            Each index i,j represents the similarities between samples i, j.
        :param n_clusters: The number of clusters to separate the data into 
        '''
        self.similarity_matrix = similarity_matrix
        self.n_clusters = n_clusters
    
    def _calculate_laplacian(self):
        D_values = self.similarity_matrix.sum(axis = 1)
        D_strength = np.diag(D_values)

        self.laplacian = D_strength - self.similarity_matrix

        return self

    def _get_transformation_matrix(self):
        _, eigvecs = np.linalg.eig(self.laplacian)
        plt.plot(_)
        print(np.around(_))
        plt.show()


        self.transformation_matrix = eigvecs[:,:self.n_clusters]


        return self

    def assign_classes(self) -> np.ndarray:
        '''
        Method to assign labels to the data givens
        '''

        plt.scatter(self.transformation_matrix[:,0], self.transformation_matrix[:,1])
        plt.show()
          
        

    
        k_means = K_means(self.transformation_matrix, self.n_clusters)
        k_means._calculate_distances()
        k_means._assign_to_cluster()
        
        return k_means.assign_classes()
        
        
        

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

    similarity_matrix = np.array(list(map(gaussian, data)))



    return similarity_matrix

if __name__ == '__main__':
    


    phi = np.linspace(-2*np.pi, 2* np.pi, 300)
    phi2 = np.linspace(-2*np.pi, 2* np.pi, 300)
    

    

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)


    x, y = pol2cart(6, phi)
    x += np.random.randn(len(x)) * 0.3
    y += np.random.randn(len(y)) * 0.3

    outer_test = np.vstack((x,y)).T

    x, y = pol2cart(2, phi2)
    x += np.random.randn(len(x)) * 0.2
    y += np.random.randn(len(y)) * 0.2

    inner_test = np.vstack((x,y)).T



    total_test = np.vstack((inner_test, outer_test))




    test_similarity = kernel_similarity(total_test)
    

    clustering = Spectral_clustering(test_similarity, 2)

    clustering._calculate_laplacian()
    clustering._get_transformation_matrix()
    classes = clustering.assign_classes()

    k_means = K_means(total_test, 2)
    k_means._calculate_distances()
    k_means._assign_to_cluster()
        
    k_means_classes = k_means.assign_classes()

    plt.figure()

    plt.scatter(total_test[:,0], total_test[:,1])



    plt.figure()
    for lab in np.unique(classes):
        ind = np.where(k_means_classes == lab)
        
        plt.scatter(total_test[ind, 0], total_test[ind, 1])


    plt.figure()
    for lab in np.unique(classes):
        ind = np.where(classes == lab)
        
        plt.scatter(total_test[ind, 0], total_test[ind, 1])

    
    plt.show()


    