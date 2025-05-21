import numpy as np
import scipy
import binning_utils
from scipy.optimize import curve_fit
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


def flatten_image(x_bin_edges_m, y_bin_edges_m, image):
    x = binning_utils.centers(x_bin_edges_m)
    y = binning_utils.centers(y_bin_edges_m)
    num_total = len(x) * len(y)
    xy = np.zeros((num_total, 2))
    w = np.zeros(num_total)
    j = 0
    for ix in range(len(x)):
        for iy in range(len(y)):
            xy[j, 0] = x[ix]
            xy[j, 1] = y[iy]
            w[j] = image[ix, iy]
            j += 1
    return xy, w


def get_distances(xy, x0, y0):
    dd = xy.copy()
    dd[:, 0] -= x0
    dd[:, 1] -= y0
    dd = dd**2
    return np.sqrt(np.sum(dd, axis=1))


def encircle_containment(
    x_bin_edges_m, y_bin_edges_m, image, x_m, y_m, quantile
):
    assert 0.0 < quantile <= 1.0
    dxy = np.mean([np.gradient(x_bin_edges_m), np.gradient(y_bin_edges_m)])
    rxy = dxy / 2
    xy_m, img_w = flatten_image(
        x_bin_edges_m=x_bin_edges_m,
        y_bin_edges_m=y_bin_edges_m,
        image=image,
    )
    img_r = get_distances(
        xy=xy_m,
        x0=x_m,
        y0=y_m,
    )
    img_r += rxy
    img_q = img_w / img_w.sum()
    rargs = np.argsort(img_r)

    img_rs = img_r[rargs]
    img_qs = img_q[rargs]

    arg_qunatile = np.argmin(np.abs(np.cumsum(img_qs) - quantile))
    r_quantile = img_rs[arg_qunatile]
    return r_quantile


def fit_gauss_in_image(x_bin_edges_m, y_bin_edges_m, image):
    xy, w = flatten_image(
        x_bin_edges_m=x_bin_edges_m,
        y_bin_edges_m=y_bin_edges_m,
        image=image,
    )
    w = w / np.percentile(w, 90)

    r_max = max(x_bin_edges_m) - min(x_bin_edges_m)
    r_min = 2.0 * np.mean(np.gradient(x_bin_edges_m))

    # gauss_pseudo_2d(xy, x0, y0, sigma)
    guess = [
        np.mean(x_bin_edges_m),
        np.mean(y_bin_edges_m),
        np.mean([r_min, r_max]),
    ]

    bounds = (
        [min(x_bin_edges_m), min(y_bin_edges_m), r_min],
        [max(x_bin_edges_m), max(y_bin_edges_m), r_max],
    )

    predicted_params, uncert_cov = curve_fit(
        f=gauss_pseudo_2d,
        xdata=xy,
        ydata=w,
        p0=guess,
        bounds=bounds,
    )
    return predicted_params


def gauss_pseudo_2d(xy, x0, y0, sigma):
    dx = xy[:, 0] - x0
    dy = xy[:, 1] - y0
    dd = np.hypot(dx, dy)
    return gauss1d(x=dd, x0=0, sigma=sigma)


def gauss1d(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / (2 * sigma**2))
