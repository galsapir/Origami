"""
Origami Plot Library

This module provides functions to create origami plots for visualizing multi-dimensional data.
"""

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from enum import Enum

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

def create_origami_plot(data: pd.Series, variable_configs: List[VariableConfig], 
                        color_scheme: ColorScheme = ColorScheme.DEFAULT, 
                        title: str = "", figsize: Tuple[int, int] = DEFAULT_FIGURE_SIZE) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
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

    # Plot data
    max_values = np.array([config.max_value for config in variable_configs])
    min_values = np.array([config.min_value for config in variable_configs])
    data_values = np.array([data[config.name] for config in variable_configs])
    
    scaled_values = (data_values - min_values) / (max_values - min_values)
    
    x_points = np.zeros(2 * n_variables)
    y_points = np.zeros(2 * n_variables)
    x_points[0::2] = x_coordinates * scaled_values
    y_points[0::2] = y_coordinates * scaled_values

    auxiliary_offset = 0.05  # Ensure the auxiliary points are visible
    x_points[1::2] = x_auxiliary_coordinates * auxiliary_offset
    y_points[1::2] = y_auxiliary_coordinates * auxiliary_offset

    # Close the polygon
    x_points = np.append(x_points, x_points[0])
    y_points = np.append(y_points, y_points[0])
    
    ax.plot(x_points, y_points, marker='o', linestyle='-', color=color_scheme.value[0], linewidth=2, markersize=6)
    ax.fill(x_points, y_points, alpha=0.1, color=color_scheme.value[1])

    # Add labels
    for config, x, y in zip(variable_configs, x_coordinates * 1.2, y_coordinates * 1.2):
        ax.text(x, y, config.name, ha='center', va='center', fontsize=8)

    plt.tight_layout()
    return fig, ax
