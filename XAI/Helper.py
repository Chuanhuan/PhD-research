#%%
import numpy as np
import matplotlib.pyplot as plt

def cluster_mask(img, means, covs, threshold=0.1):
    # Generate a grid of points for the canvas
    x, y = np.mgrid[0:28, 0:28]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    # Calculate the probability density at each point on the grid for each component
    densities = []
    for mean, cov in zip(means, covs):
        z = np.exp(-0.5 * np.sum((pos - mean) @ np.linalg.inv(cov) * (pos - mean), axis=2))
        densities.append(z)

    # Normalize the densities for each component using softmax
    # densities = np.stack(densities, axis=2)
    # densities = np.exp(densities - np.max(densities, axis=2, keepdims=True))
    # densities /= np.sum(densities, axis=2, keepdims=True)

    # Create separate masks for each component
    masks = []
    for density in densities:
        mask = density < threshold
        masks.append(mask)

    # Combine the masks using logical operations
    mask = np.stack(masks, axis=2)
    mask = np.all(mask, axis=2)

    # Create a mask where the density is above the threshold
    indeces_1 = np.where(mask)
    img[indeces_1] = np.random.rand(*indeces_1[0].shape)

    # Plot the masks using the modified image
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title("Mixture of Gaussians for Cluster Mask")
    plt.show()
    return img

#%%
# def plot_cluster_region(m, s2):
#     """
#     Plot the cluster region for each Gaussian component on a 28x28 canvas.

#     Parameters:
#         m (numpy.ndarray): The mean parameters of the Gaussian components, of shape (k, 2), where k is the number of components.
#         s2 (numpy.ndarray): The variance parameters of the Gaussian components, of shape (k, 1), where k is the number of components.

#     Returns:
#         None

#     This function plots the cluster region for each Gaussian component on a 28x28 canvas. It iterates over each component and generates points on the ellipse corresponding to the standard deviation of the component. The ellipse is then plotted using `plt.imshow()`. The mean parameters of the components are also plotted using `plt.scatter()`. The title of the plot is set to 'Cluster region'. The plot is displayed using `plt.show()`.

#     Note: The function assumes that `m` and `s2` have the same number of rows.
#     """
#     img = np.zeros((28, 28))
#     for k_idx in range(m.shape[0]):
#         angle = np.linspace(0, 2*np.pi, 100)
#         x = np.sqrt(s2[k_idx, 0]) * np.cos(angle) + m[k_idx, 0]
#         y = np.sqrt(s2[k_idx, 1]) * np.sin(angle) + m[k_idx, 1]
#         x = np.clip(x, 0, 27).astype(int)
#         y = np.clip(y, 0, 27).astype(int)
#         img[y, x] = 1
#         # Add a 1px border around each cluster region
#         # to make them more visible
#         for dy in range(-1, 2):
#             for dx in range(-1, 2):
#                 # Calculate the new indices after wrapping around
#                 # the image boundaries
#                 new_y = (y + dy) % 28
#                 new_x = (x + dx) % 28
#                 # Set the new indices to 1
#                 if new_y.any() > 25 or new_x.any() > 25:
#                     print(f"new_x[{new_x}] > 25 at index {k_idx}")
#                     print(f"new_y[{new_y}] > 25 at index {k_idx}")
#                 # assert new_y.any()>25 and new_x.any()>25
#                 img[new_y, new_x] = 1
#     plt.imshow(img)
#     plt.scatter(m[:, 0], m[:, 1], c='r')
#     plt.title('Cluster region')
#     plt.savefig('cluster_region.png')
#     plt.show()

# m_test = np.array([[22, 1], [8, 8], [15, 15]])
# s2_test = np.array([[1,3], [19,5], [2,9]])
# plot_cluster_region(m_test, s2_test)


#%%
# def cluster_mask(img, mean, cov, threshold=0.1):
#     # Generate a grid of points for the canvas
#     x, y = np.mgrid[0:28, 0:28]
#     pos = np.empty(x.shape + (2,))
#     pos[:, :, 0] = x
#     pos[:, :, 1] = y

#     # Calculate the probability density at each point on the grid
#     z = np.exp(-0.5 * np.sum((pos - mean) @ np.linalg.inv(cov) * (pos - mean), axis=2))

#     # Define the threshold for the oval boundary
#     # TODO: Find a better threshold
#     # threshold = 0.1
#     # Calculate the standard deviation for each component
#     # std_dev = np.sqrt(np.diag(cov))

#     # # Calculate the threshold for the oval boundary
#     # threshold = mean + std_dev
#     # # Convert the threshold to multivariate normal distribution pdf
#     # threshold = np.exp(-0.5 * np.sum(((pos - threshold) @ np.linalg.inv(cov) * (pos - threshold)), axis=2)) / (2 * np.pi * np.sqrt(np.prod(np.diag(cov))))

#     # Create a mask where the density is above the threshold
#     # TODO: the indences for the mask need to have the same shape as the image
#     indeces_1 = np.where(z < threshold)
#     img[indeces_1] = np.random.rand(*indeces_1[0].shape)

#     # rnd = np.random.uniform(size=indeces_1[0].shape)

#     # # TODO: decide the threshold for the mask
#     # indeces_2 = np.where(rnd < threshold)

#     # img[indeces_2] = np.random.uniform(img.min(), img.max(), indeces_2[0].shape)
#     # mask = z > threshold

#     # Plot the oval using the mask
#     plt.imshow(img, cmap=plt.cm.gray)
#     plt.title("Oval for Multivariate Normal Distribution")
#     plt.show()
#     return img


if __name__ == "__main__":
    # Define the mean and covariance matrix for the multivariate normal distribution
    mean = [[14, 14], [8, 8], [22, 15]]  # Center of the oval
    cov = [[[10, 3], [3, 20]], [[10, 3], [3, 20]], [[10, 3], [3, 20]]]  # Covariance matrix (adjust for elliptical shape)
    img = np.zeros((28, 28))
    cluster_mask(img, mean, cov)

# %%