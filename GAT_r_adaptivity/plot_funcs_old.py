"""
Functions for data visualization and logging
"""

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import datetime
import torch
import networkx as nx
import hashlib
from collections import Counter
import os

def density_plot_matrix(matrix: np.ndarray, output: str = "", title: str = "", show: bool = True) -> None:
    """
    Plot KDE (Kernel Density Estimate), used to get a rough understanding of how
    the vertexes are spaced.
    Args:
        matrix (np.ndarray): The input matrix (e.g., node coordinates).
        output (str): Directory to save the plot.
        title (str): Custom title for the plot. If empty, a default title is used.
        show (bool): Whether to display the plot interactively.
    """
    if not isinstance(matrix, np.ndarray):
        try:
            matrix = np.array(matrix)
        except Exception as e:
            print(f"Error converting matrix to NumPy array in density_plot_matrix: {e}")
            return
    if matrix.ndim == 1: # If it's already flat (e.g. for 1D coordinates)
        data = matrix
    elif matrix.ndim == 2:
        data = matrix.flatten()
    else:
        print(f"Error: density_plot_matrix expects a 1D or 2D matrix, got {matrix.ndim}D.")
        return
    
    if data.size == 0:
        print("Warning: Empty data provided to density_plot_matrix. Skipping plot.")
        return

    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 1000)  # X values for the density plot
    density = kde(x_vals)  # Evaluate density function
    
    plt.figure() # Ensure a new figure for each plot
    plt.plot(x_vals, density, color='blue', linewidth=2, label="Density")
    plt.xlabel("Value")
    plt.ylabel("Density")

    plot_title = title if title else f"Density Plot of Matrix Values | n_points = {data.size}"
    if title and hasattr(matrix, 'shape') and matrix.ndim == 2 : # Add n_rows if not part of custom title
        plot_title += f" (from matrix shape: {matrix.shape})"

    plt.title(plot_title)
    plt.grid(True)
    plt.legend()
    
    if show:
        plt.show()
    
    if output != "":
        # Ensure the output directory exists
        os.makedirs(output, exist_ok=True) 
        time_str = datetime.datetime.now().strftime("%m%d%Y_%H_%M_%S") # Added seconds
        # Sanitize title for filename
        safe_title_part = "".join([c if c.isalnum() else "_" for c in title.split(" (")[0][:30]]) if title else "density_plot"
        filename = os.path.join(output, f"{safe_title_part}_{time_str}.png")
        try:
            plt.savefig(filename)
            print(f"Saved density plot to {filename}")
        except Exception as e:
            print(f"Error saving density plot: {e}")
    plt.close() # Close the figure after showing/saving to free memory

def loss_plot(epoch_count : list[int], 
              loss_values : list[float], 
              test_loss_values : list[float],
              model_name : str = "",
              output : str = "",
              show : bool = True) -> None:
    """"
    Plot diagnostics, loss curves
    """
    plt.plot(epoch_count, loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.title(f"Training and test loss curves for model {model_name}")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.ylim(0,None)
    plt.grid(True)
    plt.legend()
    if show:
        plt.show()
    if output != "":
        time = datetime.datetime.now().strftime("%m%d%Y_%H_%M")
        plt.savefig(output + "/" + model_name + "_" + loss_plot.__name__ + "_" + time + ".png")

def predVStrue(label_val_true: list,      # list of np.arrays
               label_val_pred: list,      # list of np.arrays
               label_train_true: list,    # list of np.arrays
               label_train_pred: list,    # list of np.arrays
               model_name: str,
               output: str = "",
               show: bool = True) -> None:
    """
    Plot Predictions vs True Values (Training and/or Validation).
    Args:
        label_val_true (list): List of numpy arrays containing true validation values.
        label_val_pred (list): List of numpy arrays containing predicted validation values.
        label_train_true (list): List of numpy arrays containing true training values.
        label_train_pred (list): List of numpy arrays containing predicted training values.
        model_name (str): Name of the model for titling and saving.
        output (str): Directory to save the plot.
        show (bool): Whether to display the plot interactively.
    """
    plt.figure(figsize=(8, 6)) # Create a new figure for this plot

    plotted_training = False
    if label_train_true and label_train_pred: # Check if both lists are non-empty
        try:
            train_true_concat = np.concatenate(label_train_true)
            train_pred_concat = np.concatenate(label_train_pred)
            # Further check if the concatenated arrays are not empty
            if train_true_concat.size > 0 and train_pred_concat.size > 0:
                plt.scatter(train_true_concat, train_pred_concat, label='Training Data', alpha=0.5, s=10, c='blue')
                plotted_training = True
        except ValueError as e:
            print(f"Warning: Could not process training data for predVStrue plot: {e}")

    plotted_validation = False
    if label_val_true and label_val_pred: # Check if both lists are non-empty
        try:
            val_true_concat = np.concatenate(label_val_true)
            val_pred_concat = np.concatenate(label_val_pred)
            # Further check if the concatenated arrays are not empty
            if val_true_concat.size > 0 and val_pred_concat.size > 0:
                plt.scatter(val_true_concat, val_pred_concat, label='Validation Data', alpha=0.5, s=10, c='orange')
                plotted_validation = True
        except ValueError as e:
            print(f"Warning: Could not process validation data for predVStrue plot: {e}")

    if not plotted_training and not plotted_validation:
        print("Warning: No data provided to plot in predVStrue. Skipping plot generation.")
        plt.close() # Close the empty figure
        return

    # Determine overall min/max for the ideal line if any data was plotted
    all_true_for_line = []
    if plotted_training:
        all_true_for_line.append(train_true_concat)
    if plotted_validation:
        all_true_for_line.append(val_true_concat)
    
    if all_true_for_line: # If at least one set of data was plotted
        combined_true_for_line = np.concatenate(all_true_for_line)
        if combined_true_for_line.size > 0:
            min_val = np.min(combined_true_for_line)
            max_val = np.max(combined_true_for_line)
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal', lw=1.5)

    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'Predictions vs True Values ({model_name})', fontsize=14)
    if plotted_training or plotted_validation: # Only show legend if something was plotted
        plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if show:
        plt.show()
    
    if output != "" and isinstance(output, str):
        os.makedirs(output, exist_ok=True)
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Changed time format slightly
        safe_model_name = "".join([c if c.isalnum() else "_" for c in model_name.replace(" ", "_")[:30]])
        filename = os.path.join(output, f"{safe_model_name}_predVStrue_{time_str}.png")
        try:
            plt.savefig(filename)
            print(f"Saved predVStrue plot to {filename}")
        except Exception as e:
            print(f"Error saving predVStrue plot: {e}")
    plt.close() # Close the figure after saving/showing

def cuda_status(device : torch.device) -> None:
    """
    Used to see the current status of the GPU card
    """
    print(f"Using device: {device}")
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

def plot_accuracy_vs_cost(classical_costs, classical_accuracies, gat_costs, gat_accuracies, cost_label='Computational Cost (e.g., Nodes)', accuracy_label='FE Solution Error (e.g., L2 Norm)', title='Accuracy vs. Computational Cost Comparison'):
    """
    MADE BY GEMINI

    Generates a scatter plot comparing accuracy vs. computational cost for two methods.

    Args:
        classical_costs (list or np.array): List of computational costs for the classical method.
        classical_accuracies (list or np.array): List of corresponding accuracy metrics for the classical method.
        gat_costs (list or np.array): List of computational costs for the GAT-based method.
        gat_accuracies (list or np.array): List of corresponding accuracy metrics for the GAT-based method.
        cost_label (str): Label for the x-axis.
        accuracy_label (str): Label for the y-axis (lower is typically better).
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 6))

    # Scatter plot for individual runs
    plt.scatter(classical_costs, classical_accuracies, label='Classical r-adaptivity', marker='o', s=80, alpha=0.7)
    plt.scatter(gat_costs, gat_accuracies, label='GAT-based Optimization', marker='x', s=80, alpha=0.7)

    # Optional: Add lines if the points represent sequential refinements
    # plt.plot(sorted(classical_costs), [y for _, y in sorted(zip(classical_costs, classical_accuracies))], linestyle='--', alpha=0.5)
    # plt.plot(sorted(gat_costs), [y for _, y in sorted(zip(gat_costs, gat_accuracies))], linestyle='--', alpha=0.5)

    plt.xlabel(cost_label)
    plt.ylabel(accuracy_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Optional: Use log scale if ranges are large
    # plt.xscale('log')
    # plt.yscale('log')
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# classical_costs_data = 
# classical_accuracy_data = [0.1, 0.05, 0.02, 0.01]
# gat_costs_data =  # Example: GAT might have slightly different node counts
# gat_accuracy_data = [0.09, 0.045, 0.022, 0.011]
# plot_accuracy_vs_cost(classical_costs_data, classical_accuracy_data, gat_costs_data, gat_accuracy_data)
# ---------------------


def plot_time_comparison(classical_times, gat_times,
                         time_label='Mesh Optimization Time (s)',
                         title='Mesh Optimization Step Time Comparison',
                         use_box_plot=False,
                         output: str = "",  # Added output parameter
                         show: bool = True):   # Added show parameter
    """
    Generates a bar chart or box plot comparing the computational time
    specifically for the mesh optimization step of two methods.

    Args:
        classical_times (list, np.array, or float/int): Times for the classical method.
        gat_times (list, np.array, or float/int): Times for the GAT-based method.
        time_label (str): Label for the y-axis.
        title (str): Title for the plot.
        use_box_plot (bool): If True, generates a box plot.
        output (str): Directory to save the plot.
        show (bool): Whether to display the plot interactively.
    """
    plt.figure(figsize=(7, 6)) # Adjusted figure size
    methods = ["Classical Method", "GAT Method"]

    if use_box_plot:
        # Ensure data is list-like for boxplot
        c_times_plot = [classical_times] if np.isscalar(classical_times) else list(classical_times)
        g_times_plot = [gat_times] if np.isscalar(gat_times) else list(gat_times)

        if len(c_times_plot) < 2 or len(g_times_plot) < 2:
            print("Warning: Box plot is generally more informative with multiple data points per method.")

        data_to_plot = [c_times_plot, g_times_plot]
        bp = plt.boxplot(data_to_plot, labels=methods, patch_artist=True, vert=True, widths=0.6)

        # Basic styling for box plot
        colors = ['skyblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)
    else:  # Bar chart
        # np.mean handles single float/int inputs correctly, returning them as is.
        avg_classical_time = np.mean(classical_times)
        avg_gat_time = np.mean(gat_times)
        times_to_plot = [avg_classical_time, avg_gat_time]

        bars = plt.bar(methods, times_to_plot, color=['skyblue', 'lightcoral'], width=0.6)

        # Add text labels for average times on bars
        for bar_item in bars:
            yval = bar_item.get_height()
            # Add a small offset for the text based on the max value
            offset = (0.01 * max(times_to_plot, default=1)) if times_to_plot else 0.01
            plt.text(bar_item.get_x() + bar_item.get_width()/2.0, yval + offset,
                     f'{yval:.2e}', ha='center', va='bottom', fontsize=9)

    plt.ylabel(time_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if show:
        plt.show()

    if output != "" and isinstance(output, str):
        os.makedirs(output, exist_ok=True) # Ensure output directory exists
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # More standard time format
        # Create a safe filename from the title
        safe_title_part = "".join([c if c.isalnum() else "_" for c in title.replace(" ", "_")[:40]])
        filename = os.path.join(output, f"{safe_title_part}_time_comp_{time_str}.png")
        try:
            plt.savefig(filename)
            print(f"Saved time comparison plot to {filename}")
        except Exception as e:
            print(f"Error saving time comparison plot: {e}")
    plt.close() # Close the figure after saving/showing

# --- Example Usage ---
# Example 1: Bar chart with average times
# classical_avg_time = 150.5
# gat_avg_time = 0.8
# plot_time_comparison([classical_avg_time], [gat_avg_time], use_box_plot=False)

# Example 2: Box plot with multiple run times
# classical_run_times = 
# gat_run_times = [0.75, 0.85, 0.80, 0.78, 0.82]
# plot_time_comparison(classical_run_times, gat_run_times, use_box_plot=True)
# ---------------------


def plot_convergence(classical_dofs, classical_errors, gat_dofs, gat_errors, dof_label='Degrees of Freedom (DoF)', error_label='FE Solution Error (e.g., L2 Norm)', title='Convergence Plot'):
    """
    MADE BY GEMINI

    Generates a line plot comparing the convergence (error vs. DoF/nodes)
    of two methods.

    Args:
        classical_dofs (list or np.array): List of degrees of freedom (or node counts)
                                           for the classical method runs.
        classical_errors (list or np.array): List of corresponding FE solution errors
                                             for the classical method.
        gat_dofs (list or np.array): List of degrees of freedom (or node counts)
                                     for the GAT-based method runs.
        gat_errors (list or np.array): List of corresponding FE solution errors
                                       for the GAT-based method.
        dof_label (str): Label for the x-axis.
        error_label (str): Label for the y-axis.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 6))

    # Sort data points by DoF for plotting lines correctly
    classical_sorted_indices = np.argsort(classical_dofs)
    gat_sorted_indices = np.argsort(gat_dofs)

    plt.plot(np.array(classical_dofs)[classical_sorted_indices], np.array(classical_errors)[classical_sorted_indices],
             label='Classical r-adaptivity', marker='o', linestyle='-')
    plt.plot(np.array(gat_dofs)[gat_sorted_indices], np.array(gat_errors)[gat_sorted_indices],
             label='GAT-based Optimization', marker='x', linestyle='-')

    plt.xlabel(dof_label)
    plt.ylabel(error_label)
    plt.xscale('log') # Common practice for convergence plots
    plt.yscale('log') # Common practice for convergence plots
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# classical_dofs_data = 
# classical_error_data = [0.1, 0.05, 0.02, 0.01, 0.005]
# gat_dofs_data =  # Example: GAT might have slightly different DoFs
# gat_error_data = [0.09, 0.045, 0.022, 0.011, 0.0055]
# plot_convergence(classical_dofs_data, classical_error_data, gat_dofs_data, gat_error_data)
# ---------------------


def plot_gat_training_loss(epochs, training_loss, validation_loss=None, loss_label='Loss', title='GAT Model Training Progress'):
    """
    MADE BY GEMINI

    (Optional) Generates a line plot showing the GAT model's training
    and optionally validation loss over epochs.

    Args:
        epochs (list or np.array): List of epoch numbers (e.g., range(num_epochs)).
        training_loss (list or np.array): List of training loss values per epoch.
        validation_loss (list or np.array, optional): List of validation loss values per epoch.
                                                     Defaults to None.
        loss_label (str): Label for the y-axis.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 6))

    plt.plot(epochs, training_loss, label='Training Loss', linestyle='-', color='blue')

    if validation_loss is not None:
        plt.plot(epochs, validation_loss, label='Validation Loss', linestyle='--', color='orange')

    plt.xlabel('Epoch')
    plt.ylabel(loss_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Optional: Use log scale if loss decreases rapidly
    # plt.yscale('log')
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
# num_epochs = 100
# epochs_data = range(1, num_epochs + 1)
# # Example loss data (replace with your actual data)
# train_loss_data = [10 / (epoch**0.5) for epoch in epochs_data]
# val_loss_data = [12 / (epoch**0.5) + np.random.rand() * 0.1 for epoch in epochs_data] # Add some noise
# plot_gat_training_loss(epochs_data, train_loss_data, validation_loss=val_loss_data)
# ---------------------

def weisfeiler_lehman_test(G, iterations=3):
    """
    Performs the Weisfeiler-Lehman test (1-WL) on a graph.

    Args:
        G (nx.Graph): The graph to analyze (NetworkX object).
        iterations (int): The number of WL iterations to perform.

    Returns:
        list: A list where each element is a Counter object representing
              the histogram (distribution) of node labels for that iteration.
              The list includes the histogram for iteration 0 (initial) up to
              the final iteration.
    """
    # --- Initialization (Iteration 0) ---
    # Using node degree as the initial label. Other attributes could also be used.
    node_labels = {node: str(G.degree(node)) for node in G.nodes()}
    label_history = [] # List to store label distributions per iteration
    label_history.append(Counter(node_labels.values()))

    print(f"Iteration 0 (Initial): {len(label_history[0])} unique labels")
    # print(f"  Labels: {dict(label_history[0])}") # Uncomment for details

    # --- WL Iterations ---
    for k in range(iterations):
        new_labels = {}
        aggregate_signatures = {} # Intermediate storage for signatures before hashing

        for node in G.nodes():
            # 1. Aggregate neighbor labels
            neighbor_labels = sorted([node_labels[neighbor] for neighbor in G.neighbors(node)])

            # 2. Create signature: (own_label, sorted_neighbor_labels)
            signature = (node_labels[node], tuple(neighbor_labels))

            # Store the signature temporarily
            aggregate_signatures[node] = signature

        # 3. Map unique signatures to new, compressed labels (via hashing)
        unique_signatures = sorted(list(set(aggregate_signatures.values())))
        signature_to_compressed_label = {}
        for i, sig in enumerate(unique_signatures):
            # Use a stable hash to create the new label
            # We hash the string representation of the signature
            hash_object = hashlib.sha256(str(sig).encode())
            # Convert hash to a shorter string/int if desired, here we use the first 8 chars
            compressed_label = hash_object.hexdigest()[:8]
            # Alternatively, just use a simple counter: compressed_label = str(i)
            signature_to_compressed_label[sig] = compressed_label

        # 4. Assign the new, compressed labels to the nodes
        for node in G.nodes():
            new_labels[node] = signature_to_compressed_label[aggregate_signatures[node]]

        # Update labels for the next round
        node_labels = new_labels

        # Store the histogram for this iteration
        current_histogram = Counter(node_labels.values())
        label_history.append(current_histogram)

        print(f"Iteration {k+1}: {len(current_histogram)} unique labels")
        # print(f"  Labels: {dict(current_histogram)}") # Uncomment for details

        # Check for convergence (optional): If the histogram is the same as the previous one
        if k > 0 and current_histogram == label_history[-2]:
            print(f"Converged after {k+1} iterations.")
            break

    return label_history

def plot_wl_histograms(label_history, graph_name="Graph"):
    """
    Plots histograms of node labels for each WL iteration.

    Args:
        label_history (list): The list of Counter objects from weisfeiler_lehman_test.
        graph_name (str): The name of the graph for the plot title.
    """
    num_iterations = len(label_history)
    fig, axes = plt.subplots(1, num_iterations, figsize=(5 * num_iterations, 4), sharey=True)
    if num_iterations == 1: # Matplotlib doesn't return an array for a single subplot
        axes = [axes]

    fig.suptitle(f'Weisfeiler-Lehman Label Distribution for {graph_name}', fontsize=16)

    # Find all unique labels across all iterations for consistent plotting
    unique_labels_across_iterations = set()
    for hist in label_history:
        unique_labels_across_iterations.update(hist.keys())

    # Sort labels for consistent plotting order (can be numeric or alphabetical)
    sorted_unique_labels = sorted(list(unique_labels_across_iterations))
    # Map labels to indices for plotting
    label_to_index = {label: i for i, label in enumerate(sorted_unique_labels)}
    num_unique_labels = len(sorted_unique_labels)

    for i, hist in enumerate(label_history):
        ax = axes[i]
        # Get counts for each unique label, defaulting to 0 if not present
        counts = [hist.get(label, 0) for label in sorted_unique_labels]
        indices = np.arange(num_unique_labels) # x-coordinates for the bars

        ax.bar(indices, counts, tick_label=sorted_unique_labels)
        ax.set_title(f'Iteration {i}')
        ax.set_xlabel('Node Label')
        if i == 0:
            ax.set_ylabel('Number of Nodes')
        ax.tick_params(axis='x', rotation=90) # Rotate labels if they are long

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap with suptitle
    plt.show()

