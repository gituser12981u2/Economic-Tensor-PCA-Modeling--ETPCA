# src/tensor_model.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def normalize_data(dataframe, column):
    """ Normalizes the specified column of the dataframe. """
    scaler = MinMaxScaler()
    return scaler.fit_transform(dataframe[[column]])


def tensor_product(tensor1, tensor2):
    """ Computes the tensor product of two tensors. """
    return np.tensordot(tensor1, tensor2, axes=0)


def process_tensors(djia_df, cpi_df, gnp_df):
    """ Processes the dataframes and returns the tensor product. """
    tensor_djia = normalize_data(djia_df, 'DJIA')
    tensor_cpi = normalize_data(cpi_df, 'CPI')
    tensor_gnp = normalize_data(gnp_df, 'GNP_Predicted')

    # First, combine CPI and DJIA
    tensor_product_cpi_djia = np.tensordot(tensor_cpi, tensor_djia, axes=0)

    # Then involve GNP
    tensor_product_with_gnp = np.tensordot(
        tensor_product_cpi_djia, tensor_gnp, axes=0)

    # Reshape and apply PCA
    tensor_product_with_gnp = tensor_product_with_gnp.reshape(
        tensor_product_with_gnp.shape[0], -1)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(tensor_product_with_gnp)

    return principal_components
