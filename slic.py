import numpy as np


class SLIC():
    """
    
    """
    def __init__(self, img, n_clusters) -> None:
        self.img = img
        self.n_clusters = n_clusters 



    def _initiate_clusters(self):
        n_pixels_in_img = self.img.shape[0] * self.img.shape[1]

        grid_spacing_parameter = int((n_pixels_in_img / self.n_clusters) ** (1/2))

        posisiton_x = np.arange(grid_spacing_parameter, self.img.shape[0] - grid_spacing_parameter, grid_spacing_parameter)
        position_y = np.arange(grid_spacing_parameter, self.img.shape[1] - grid_spacing_parameter, grid_spacing_parameter)
        
        cluster_position_x, cluster_position_y = np.array(np.meshgrid(posisiton_x, position_y))
        
        find_3x3 = lambda array, x, y: array[x-1: x+2, y-1:y+2]
        centered_3x3_indices = np.indices((3,3)) - 1

        #rgb_cluster_values = self.img[cluster_positions_x, cluster_positions_y]


        for (x, y)in zip(cluster_position_x.flatten(), cluster_position_y.flatten()):
            
            for neighbourhood_x, neighbourhood_y in zip(centered_3x3_indices[0].flatten(), centered_3x3_indices[1].flatten()):
                print(neighbourhood_x, neighbourhood_y)
                neighbourhood_values = find_3x3(self.img, x - neighbourhood_x, y - neighbourhood_y)
                print(neighbourhood_values)


        #self.n_cluster_vectors = np.dstack((rgb_cluster_values, cluster_positions_x, cluster_positions_y))
        #print(self.n_cluster_vectors.shape)


    def _add_spatial_positions_to_img(self):
        """
        method to add spatial positional values to each rgb vector
        in the image.
        """
        indices_x, indices_y = np.indices((self.img.shape[0], self.img.shape[1]))

        print(indices_x.shape, indices_y.shape, self.img.shape)
        self.extended_img = np.dstack((self.img, indices_x, indices_y))

        return self
    

from dependencies import *


if __name__ == "__main__":


    images = import_images('images')


    a = SLIC(images[1], 1000)
    a._add_spatial_positions_to_img()
    a._initiate_clusters()