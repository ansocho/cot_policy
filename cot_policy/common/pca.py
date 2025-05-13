import torch


class PCA:
    def __init__(self, n_components):
        """
        PCA Transformer class for training and transforming data.

        Args:
            n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
        self.V = None  # Placeholder for the principal components

    def fit(self, data):
        """
        Fit the PCA model to the data and compute principal components.

        Args:
            data (torch.Tensor): Input data of shape (batch_size, features).
        """
        if self.n_components > data.shape[0]:
            self.n_components = data.shape[0]

        # Center the data
        data_centered = data - data.mean(dim=0, keepdim=True)

        # Perform SVD to compute principal components
        _, _, V = torch.svd_lowrank(data_centered, q=self.n_components)

        # Retain the top n_components principal components
        self.V = V
        # self.S = S
        # Calculate explained variance
        # self.total_variance = (S**2).sum().item()  # Total variance

    def transform(self, data, n_components=None):
        """
        Transform the data using the trained principal components.

        Args:
            data (torch.Tensor): Input data of shape (batch_size, features).

        Returns:
            torch.Tensor: Transformed data of shape (batch_size, n_components).
        """
        if n_components is None:
            n_components = self.n_components
        if self.V is None:
            raise RuntimeError("The PCA model has not been trained. Call `fit` first.")

        # Center the data
        data_centered = data - data.mean(dim=0, keepdim=True)

        # print(
        #     f"Explained variance with {n_components} components: {self.variance_ratio(n_components)}"
        # )

        # Project data onto the principal components
        return torch.mm(data_centered, self.V[:, :n_components])

    def fit_transform(self, data):
        """
        Fit the PCA model to the data and transform it in one step.

        Args:
            data (torch.Tensor): Input data of shape (batch_size, features).

        Returns:
            torch.Tensor: Transformed data of shape (batch_size, n_components).
        """
        self.fit(data)
        return self.transform(data)

    def variance_ratio(self, n_components):
        """
        Return the explained variance of the top n_components principal components.
        """
        return (self.S[:n_components] ** 2).sum().item() / self.total_variance
