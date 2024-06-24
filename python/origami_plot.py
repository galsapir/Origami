"""
Origami Plot Library

This module provides functions to create origami plots for visualizing multi-dimensional data.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

# Constants
DEFAULT_GRID_LINES = 4
DEFAULT_FIGURE_SIZE = (10, 10)

@dataclass
class VariableConfig:
    """Configuration for each variable in the origami plot."""
    name: str
    min_value: float
    max_value: float

class ColorScheme(Enum):
    """Color schemes for the origami plot."""
    DEFAULT = ((0.2, 0.5, 0.5, 1), (0.2, 0.5, 0.5, 0.1))
    COLORBLIND_FRIENDLY = ((0.0, 0.4470, 0.7410, 1), (0.0, 0.4470, 0.7410, 0.1))


def create_origami_plot(data_series: List[pd.Series], variable_configs: List[VariableConfig], 
                        color_scheme: ColorScheme = ColorScheme.DEFAULT, 
                        title: str = "", figsize: Optional[Tuple[int, int]] = None,
                        ax: Optional[plt.Axes] = None) -> Tuple[Optional[plt.Figure], plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGURE_SIZE)
    else:
        fig = None

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, pad=20)

    n_variables = len(variable_configs)
    theta = np.linspace(0, 2 * np.pi, n_variables, endpoint=False)
    auxiliary_theta = theta + (np.pi / n_variables)

    x_coordinates = np.cos(theta)
    y_coordinates = np.sin(theta)
    x_auxiliary_coordinates = np.cos(auxiliary_theta)
    y_auxiliary_coordinates = np.sin(auxiliary_theta)

    # Draw main axes
    for x, y in zip(x_coordinates, y_coordinates):
        ax.plot([0, x], [0, y], color='gray', alpha=0.3, linewidth=1, linestyle='-')

    # Draw auxiliary axes
    for x, y in zip(x_auxiliary_coordinates, y_auxiliary_coordinates):
        ax.plot([0, x], [0, y], color='gray', alpha=0.3, linewidth=1, linestyle='--')

    # Draw grid
    for i in range(1, DEFAULT_GRID_LINES + 1):
        circle = plt.Circle((0, 0), i / DEFAULT_GRID_LINES, fill=False, linestyle=':', linewidth=0.5, color='black')
        ax.add_artist(circle)
        ax.text(-0.05, i / DEFAULT_GRID_LINES, f'{i / DEFAULT_GRID_LINES:.2f}', ha='right', va='center', color='gray')

    # Generate a color palette
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data_series)))

    # Plot data for each series
    for i, (data, color) in enumerate(zip(data_series, colors)):
        opacity = 1 if i == 0 else 0.5  # Full opacity for real patient, half for generated
        plot_data_series(ax, data, variable_configs, color, opacity, x_coordinates, y_coordinates, x_auxiliary_coordinates, y_auxiliary_coordinates)

    # Add labels
    for config, x, y in zip(variable_configs, x_coordinates * 1.2, y_coordinates * 1.2):
        ax.text(x, y, config.name, ha='center', va='center', fontsize=8)

    # Add legend
    legend_elements = [plt.Line2D([0], [0], color=colors[0], lw=2, label='Real Patient'),
                       plt.Line2D([0], [0], color=colors[1], lw=2, alpha=0.5, label='Generated Data')]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))

    if fig:
        plt.tight_layout()
    return fig, ax

def plot_data_series(ax: plt.Axes, data: pd.Series, variable_configs: List[VariableConfig], 
                     color: Tuple[float, float, float, float], opacity: float,
                     x_coordinates: np.ndarray, y_coordinates: np.ndarray,
                     x_auxiliary_coordinates: np.ndarray, y_auxiliary_coordinates: np.ndarray):
    n_variables = len(variable_configs)

    max_values = np.array([config.max_value for config in variable_configs])
    min_values = np.array([config.min_value for config in variable_configs])
    data_values = np.array([data[config.name] for config in variable_configs])
    
    scaled_values = (data_values - min_values) / (max_values - min_values)
    
    main_x = x_coordinates * scaled_values
    main_y = y_coordinates * scaled_values
    
    auxiliary_offset = 0.05  # Small offset for auxiliary points
    aux_x = x_auxiliary_coordinates * auxiliary_offset
    aux_y = y_auxiliary_coordinates * auxiliary_offset

    # Draw lines connecting main points to auxiliary points
    for i in range(n_variables):
        ax.plot([main_x[i], aux_x[i]], [main_y[i], aux_y[i]], 
                color=color, alpha=opacity, linewidth=2)

    # Draw lines connecting auxiliary points to next main points
    for i in range(n_variables):
        next_i = (i + 1) % n_variables
        ax.plot([aux_x[i], main_x[next_i]], [aux_y[i], main_y[next_i]], 
                color=color, alpha=opacity, linewidth=2)

    # Plot main points
    ax.scatter(main_x, main_y, color=color, alpha=opacity, s=30)

def create_multiple_origami_plots(df: pd.DataFrame, variable_configs: List[VariableConfig], 
                                  id_column: str = 'Patient ID', color_scheme: ColorScheme = ColorScheme.DEFAULT, 
                                  figsize: Tuple[int, int] = DEFAULT_FIGURE_SIZE,
                                  subplot_layout: Optional[Tuple[int, int]] = None) -> List[Tuple[Optional[plt.Figure], plt.Axes]]:
    plots = []
    unique_prefixes = df[id_column].str.replace(r'_[123]$', '', regex=True).unique()

    if subplot_layout:
        fig, axs = plt.subplots(*subplot_layout, figsize=(figsize[0]*subplot_layout[1], figsize[1]*subplot_layout[0]))
        axs = axs.flatten()
    else:
        fig, axs = None, [None] * len(unique_prefixes)

    for i, prefix in enumerate(unique_prefixes):
        patient_data = df[df[id_column].str.startswith(prefix)]
        data_series = [patient_data.iloc[j].drop(id_column) for j in range(len(patient_data))]
        
        fig_i, ax = create_origami_plot(
            data_series, 
            variable_configs, 
            color_scheme, 
            title=f"{id_column}: {prefix}",
            figsize=None if subplot_layout else figsize,
            ax=axs[i]
        )
        plots.append((fig_i, ax))

    if subplot_layout:
        plt.tight_layout()
        return [(fig, ax) for ax in axs]
    else:
        return plots

def create_mosaic_origami_plots(df: pd.DataFrame, variable_configs: List[VariableConfig], 
                                id_column: str = 'Patient ID', color_scheme: ColorScheme = ColorScheme.DEFAULT, 
                                mosaic_layout: Tuple[int, int] = (3, 3), 
                                figsize: Tuple[int, int] = (15, 15)) -> plt.Figure:
    fig = plt.figure(figsize=figsize)
    plots = create_multiple_origami_plots(df, variable_configs, id_column, color_scheme, subplot_layout=mosaic_layout)
    
    for (_, ax) in plots:
        if ax.get_title() == '':
            ax.remove()
    
    plt.tight_layout()
    return fig