import numpy as np


def analyse_linear_polarization_over_time(E_field_vs_time):
    """
    Returns polarization factor [0, 1] and according vector.
    """
    assert len(E_field_vs_time.shape) == 2
    assert E_field_vs_time.shape[1] == 3

    E_covarianve_matrix = np.cov(E_field_vs_time.T)
    eigen_values, eigen_vectors = np.linalg.eig(E_covarianve_matrix)

    # how strong is polarization?
    larges_eigen_value_index = np.argmax(eigen_values)

    larges_eigen_value = eigen_values[larges_eigen_value_index]
    sum_of_all_eigen_values = np.sum(eigen_values)

    factor = larges_eigen_value / sum_of_all_eigen_values
    vector = eigen_vectors[larges_eigen_value_index]
    return factor, vector


def analyse_linear_polarization(electric_fields, channel_mask):
    factors = []
    vectors = []
    for channel in range(electric_fields.num_channels):
        if channel_mask[channel]:
            factor, vector = analyse_linear_polarization_over_time(
                E_field_vs_time=electric_fields[channel]
            )
            factors.append(factor)
            vectors.append(vector)
    factors = np.array(factors)
    vectors = np.array(vectors)

    xxs = vectors[:, 0]
    yys = vectors[:, 1]
    phis = np.arctan2(yys, xxs)

    return (np.median(factors), np.std(factors)), (
        np.median(phis),
        np.std(phis),
    )
