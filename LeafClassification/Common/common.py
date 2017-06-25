#!/usr/bin/env python3
"""Common functions for printing banners and plotting charts.

Usage:

    python3 words.py <URL>
"""

import seaborn as sb

from IPython.core.pylabtools import figsize
from pylab import *

sb.set_style("dark")


def plot_chart(x_label, y_label, x_component_array, y_component_array, title_text='', sigma_component=None, dash_style='k.-', log_scale='x'):
    """Plot single function chart.

        Args:
            x_label: x axis label.
            y_label: y axis label.
            x_component_array: x axis values.
            y_component_array: y axis values.
            title_text: Chart title.
            sigma_component: x_component_array standard deviation.
            dash_style: Stile of plotted curve.
            log_scale: Log scale for x or y axis.

    """
    figure(figsize(8, 4))

    if sigma_component is None:
        plot(x_component_array, y_component_array, dash_style)
    else:
        errorbar(x_component_array, y_component_array, yerr=sigma_component, fmt=dash_style)

    if log_scale == 'x':
        xscale('log')

    if log_scale == 'y':
        yscale('log')

    xlim(0.9)
    xlabel(x_label, size=20)
    ylabel(y_label, size=20)
    grid(which='both')
    title(title_text)
    plt.show()


def plot_pca_chart(x_label, y_label, x_component_array, y_component_array, title_text=''):
    """Plot chart for PCA data variance distribution.

        Args:
            x_label: x axis label.
            y_label: y axis label.
            x_component_array: x axis values.
            y_component_array: y axis values.
            title_text: Chart title.

    """
    figure(figsize(8, 4))
    plot(x_component_array, y_component_array, 'k.-')
    xscale("log")
    ylim(9e-2, 1.1)
    yticks(linspace(0.2, 1.0, 9))
    xlim(0.9)
    grid(which="both")
    title(title_text)
    xlabel(x_label, size=20)
    ylabel(y_label, size=20)
    plt.show()


def plot_multiple_charts(x_label, y_label, x_axis_array, errorbar_two_dim_mean_array, errorbar_two_dim_std_array, legend, title_text=''):
    """Plot multiple functions on a single chart.

        Args:
            x_label: x axis label.
            y_label: y axis label.
            x_axis_array: x axis values.
            errorbar_two_dim_mean_array: Array of arrays where each array is result of different function.
            errorbar_two_dim_std_array: Array of arrays where each array represents standard deviation values
                                        from respective array in errorbar_two_dim_mean_array
            legend: Legend to be drawn in right lower corner.
            title_text: Chart title.

    """
    plt.figure()

    for index in range(len(errorbar_two_dim_mean_array)):
        plt.errorbar(x=x_axis_array, y=errorbar_two_dim_mean_array[index], yerr=errorbar_two_dim_std_array[index])

    plt.xlim(min(x_axis_array) - 1, max(x_axis_array) + 1)
    plt.xlabel(x_label, size=20)
    plt.ylabel(y_label, size=20)
    plt.title(title_text, size=20)
    plt.legend(legend, loc=4)
    plt.grid(which='both')
    plt.show()


def print_header(text):
    """Print banner with header text.

        Args:
            text: Text to be printed.

    """
    output = "*  {0}  *".format(text)
    banner = "+" + "*" * (len(output) - 2) + "+"
    border = "*" + " " * (len(output) - 2) + "*"
    lines = [banner, border, output, border, banner]
    header = '\n'.join(lines)
    print()
    print()
    print(header)
    print()
    print()


def print_single_section(title_text, data_scrubbing_description, score):
    """Print banner for single score.

        Args:
            title_text: Title text of the banner to be printed.
            data_scrubbing_description: Banner description.
            score: Single score of classification algorithm

    """
    output = "Results of classification {0}:".format(data_scrubbing_description)
    banner = "+" + '-' * (len(output) + 2) + "+"
    border = " "
    accuracy = "\tMean Accuracy is {0:.6f}".format(np.mean(score))
    lines = [banner, title_text, border, output, border, accuracy, border, banner]
    section = '\n'.join(lines)
    print(section)
    print()


def print_multiple_section(title_text, data_scrubbing_description, bullet_message, components_array, scores_array):
    """Print banner for array of scores.

        Args:
            title_text: Title text of the banner to be printed.
            data_scrubbing_description: Banner description.
            bullet_message: Message to be shown in each bullet, bullet message takes two arguments for
                            string formatting. (example: bullet_message="number {0}, score {1}")
            components_array: Array of components for which calculation was performed.
            scores_array: Array of classification scores.

    """
    output = "Results of classification {0}:".format(data_scrubbing_description)
    banner = "+" + '-' * (len(output) + 2) + "+"
    border = " "
    bullet_points = []
    i = 0

    for component in components_array:
        bullet_points.append('\t-' + bullet_message.format(component, scores_array[i]))
        i += 1

    bullets_section = "\n".join(bullet_points)

    lines = [banner, title_text, border, output, border, bullets_section, border, banner]
    section = "\n".join(lines)
    print(section)
    print()
