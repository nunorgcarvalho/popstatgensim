"""Plotting helpers for genetics analyses."""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..genetics.pca import PCAResult, _format_pc_axis_label, _validate_pca_axes


def plot_site_frequency_spectrum(freqs: np.ndarray,
                                 maf: bool = True,
                                 plot_freq: bool = True,
                                 n_alleles: int = None,
                                 bin_width: float = 0.01,
                                 ax=None,
                                 title: str = None,
                                 color: str = 'steelblue',
                                 edgecolor: str = 'black'):
    '''
    Plots a site frequency spectrum from an array of allele frequencies.
    Parameters:
        freqs (1D array): Allele frequencies between 0 and 1.
        maf (bool): If True (default), converts allele frequencies to minor-allele
            frequencies before plotting.
        plot_freq (bool): If True (default), plots allele frequencies. If False, plots
            allele counts.
        n_alleles (int): Total number of sampled alleles. Required when
            `plot_freq=False`.
        bin_width (float): Width of frequency bins when `plot_freq=True`. Default is
            0.01 (1% bins).
        ax (matplotlib axis): Optional axis to draw on. If not provided, a new figure is
            created.
        title (str): Optional plot title.
        color (str): Bar color.
        edgecolor (str): Bar edge color.
    Returns:
        matplotlib axis: Axis containing the site frequency spectrum.
    '''
    freqs = np.asarray(freqs, dtype=float)
    if freqs.ndim != 1:
        raise ValueError('freqs must be a 1D array.')
    if np.any((freqs < 0) | (freqs > 1)):
        raise ValueError('freqs must contain values between 0 and 1.')

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    if plot_freq:
        if bin_width <= 0:
            raise ValueError('bin_width must be positive when plotting frequencies.')
        values = np.minimum(freqs, 1 - freqs) if maf else freqs
        upper = 0.5 if maf else 1.0
        bins = np.arange(0.0, upper + bin_width, bin_width)
        if bins[-1] < upper:
            bins = np.append(bins, upper)
        ax.hist(values, bins=bins, color=color, edgecolor=edgecolor)
        ax.set_xlabel('Minor Allele Frequency' if maf else 'Allele Frequency')
        ax.set_xlim(0.0, upper)
    else:
        if n_alleles is None:
            raise ValueError('n_alleles must be provided when plot_freq=False.')
        n_alleles = int(n_alleles)
        if n_alleles <= 0:
            raise ValueError('n_alleles must be a positive integer.')

        counts = freqs * n_alleles
        counts_rounded = np.rint(counts)
        if not np.allclose(counts, counts_rounded, atol=1e-8, rtol=1e-8):
            raise ValueError(
                'freqs are not compatible with integer allele counts for the provided n_alleles.'
            )
        counts_rounded = counts_rounded.astype(int)
        values = np.minimum(counts_rounded, n_alleles - counts_rounded) if maf else counts_rounded
        max_count = n_alleles // 2 if maf else n_alleles
        bins = np.arange(-0.5, max_count + 1.5, 1.0)
        ax.hist(values, bins=bins, color=color, edgecolor=edgecolor)
        ax.set_xlabel('Minor Allele Count' if maf else 'Allele Count')
        ax.set_xlim(-0.5, max_count + 0.5)

    if title is None:
        title = 'Site Frequency Spectrum'
    ax.set_title(title)
    ax.set_ylabel('Number of Variants')
    return ax

def plot_PCA(pca: PCAResult, pcs: Tuple[int, int] = (1, 2),
             values: np.ndarray = None,
             categorical: Optional[bool] = None,
             ax=None, title: str = 'PCA',
             color_label: str = None,
             alpha: float = 0.8, s: float = 24.0,
             cmap: str = 'viridis'):
    '''
    Plots a pair of principal components from a `PCAResult`.
    Parameters:
        pca (PCAResult): PCA result returned by `compute_PCA()`.
        pcs (tuple): Two 1-based PCs to plot on the x- and y-axes.
        values (1D array): Optional per-individual values used to color points.
        categorical (bool): Whether `values` should be treated as categorical. If not
            provided, non-numeric values are treated as categorical and numeric values as
            continuous.
        ax (matplotlib axis): Optional axis to draw on. If not provided, a new figure is
            created.
        title (str): Plot title.
        color_label (str): Label used for the legend or colorbar.
        alpha (float): Point transparency.
        s (float): Point size.
        cmap (str): Colormap used for continuous values.
    Returns:
        matplotlib axis: Axis containing the PCA plot.
    '''
    pc_x, pc_y = _validate_pca_axes(pcs)
    if max(pc_x, pc_y) > pca.scores.shape[1]:
        raise ValueError(
            f'PCA result only contains {pca.scores.shape[1]} component(s), '
            f'but pcs={pcs} was requested.'
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    x = pca.scores[:, pc_x - 1]
    y = pca.scores[:, pc_y - 1]

    if values is None:
        ax.scatter(x, y, alpha=alpha, s=s)
    else:
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError('values must be a 1D array.')
        if values.shape[0] != pca.scores.shape[0]:
            raise ValueError('values must have the same length as the number of PCA samples.')

        if categorical is None:
            categorical = not np.issubdtype(values.dtype, np.number)

        if categorical:
            unique_values = list(dict.fromkeys(values.tolist()))
            palette = plt.get_cmap('tab20', max(len(unique_values), 1))
            for i, value in enumerate(unique_values):
                mask = values == value
                ax.scatter(
                    x[mask], y[mask],
                    alpha=alpha, s=s,
                    color=palette(i),
                    label=str(value),
                )
            ax.legend(title=color_label)
        else:
            if not np.issubdtype(values.dtype, np.number):
                raise ValueError('Continuous PCA coloring requires numeric values.')
            scatter = ax.scatter(
                x, y,
                c=values.astype(float),
                cmap=cmap,
                alpha=alpha,
                s=s,
            )
            plt.colorbar(scatter, ax=ax, label=color_label)

    ax.set_xlabel(_format_pc_axis_label(pca, pc_x))
    ax.set_ylabel(_format_pc_axis_label(pca, pc_y))
    ax.set_title(title)
    return ax
