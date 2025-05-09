import numpy as np
import scipy
from astropy.convolution.kernels import Gaussian2DKernel

from ... import utils


def make_2d_gaussian_convolution_kernel(width, std=0.2):
    return Gaussian2DKernel(
        x_stddev=width * std,
        x_size=width,
        y_size=width,
    ).array


def oversample_image_twice(image):
    oversampling = 2
    out = np.zeros(
        shape=(
            image.shape[0] * oversampling,
            image.shape[1] * oversampling,
        )
    )
    for nx in range(out.shape[0]):
        for ny in range(out.shape[1]):
            out[nx, ny] = image[nx // 2, ny // 2]
    return out


def _analyse_image(image, containment_quantile=0.8):
    num_bins_quantile = find_quantile_bins(image, q=containment_quantile)

    kernel_width = int(np.round(np.sqrt(num_bins_quantile)))
    kernel_width = np.max([3, kernel_width])
    o_kernel_width = 2 * kernel_width

    o_image = oversample_image_twice(image)
    o_kernel = make_2d_gaussian_convolution_kernel(width=o_kernel_width)
    o_smooth_image = scipy.signal.convolve2d(
        in1=o_image,
        in2=o_kernel,
        mode="same",
        boundary="fill",
        fillvalue=0.0,
    )

    o_argmax = utils.argmaxNd(o_smooth_image)

    return {
        "num_bins_quantile": num_bins_quantile,
        "argmax_x_bin": o_argmax[0] / 2,
        "argmax_y_bin": o_argmax[1] / 2,
    }


def analyse_image(
    x_bin_edges_m, y_bin_edges_m, image, containment_quantile=0.8
):
    ana = _analyse_image(
        image=image, containment_quantile=containment_quantile
    )
    bx = x_bin_edges_m
    by = y_bin_edges_m

    ccx = np.interp(x=ana["argmax_x_bin"], xp=np.arange(0, len(bx)), fp=bx)
    ccy = np.interp(x=ana["argmax_y_bin"], xp=np.arange(0, len(by)), fp=by)
    x_bin_width = np.mean(np.gradient(bx))
    y_bin_width = np.mean(np.gradient(by))
    assert 0.9 < x_bin_width / y_bin_width < 1.1

    ana["argmax_x_m"] = ccx
    ana["argmax_y_m"] = ccy
    ana["area_quantile_m2"] = ana["num_bins_quantile"] * x_bin_width**2
    ana["radius_quantile_m"] = np.sqrt(ana["area_quantile_m2"] / np.pi)

    return ana


def find_quantile_bins(x, q):
    f = np.flip(np.sort(x.flatten()))
    total = np.sum(f)
    fraction = total * q
    cumsum_f = np.cumsum(f)
    idx = np.argmin(np.abs(cumsum_f - fraction))
    return idx
