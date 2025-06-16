"""
Functions for data visualization and logging
"""

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import datetime
import torch
import networkx as nx # Keep if weisfeiler_lehman_test is used
import hashlib # Keep if weisfeiler_lehman_test is used
from collections import Counter # Keep if weisfeiler_lehman_test is used
import os

def density_plot_matrix(matrix: np.ndarray, output: str = "", title: str = "", show: bool = True, **kwargs) -> None: # Added **kwargs
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
    if matrix.ndim == 1: 
        data = matrix
    elif matrix.ndim == 2:
        data = matrix.flatten()
    else:
        print(f"Error: density_plot_matrix expects a 1D or 2D matrix, got {matrix.ndim}D.")
        return
    
    if data.size == 0:
        print("Warning: Empty data provided to density_plot_matrix. Skipping plot.")
        return

    plt.figure() 
    try:
        kde = gaussian_kde(data)
        x_vals = np.linspace(np.min(data), np.max(data), 1000)
        density = kde(x_vals) 
        plt.plot(x_vals, density, color='blue', linewidth=2, label="Density")
    except Exception as e_kde: # Handle cases where KDE might fail (e.g. all points identical)
        print(f"KDE plot failed: {e_kde}. Plotting histogram instead.")
        plt.hist(data, bins='auto', density=True, color='skyblue', label="Histogram (fallback)")

    plt.xlabel("Value")
    plt.ylabel("Density")

    plot_title = title if title else f"Density Plot of Matrix Values | n_points = {data.size}"
    if title and hasattr(matrix, 'shape') and matrix.ndim == 2 : 
        plot_title += f" (from matrix shape: {matrix.shape})"

    plt.title(plot_title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    if output != "":
        os.makedirs(output, exist_ok=True) 
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
        safe_title_part = "".join([c if c.isalnum() else "_" for c in title.split(" (")[0][:30]]) if title else "density_plot"
        filename = os.path.join(output, f"{safe_title_part}_{time_str}.png")
        try:
            plt.savefig(filename)
            print(f"Saved density plot to {filename}")
        except Exception as e:
            print(f"Error saving density plot: {e}")
    
    if show:
        plt.show()
    plt.close() 

def loss_plot(epoch_count : list[int], 
              loss_values : list[float], 
              test_loss_values : list[float],
              model_name : str = "",
              output : str = "",
              show : bool = True) -> None:
    plt.figure() # Ensure new figure
    plt.plot(epoch_count, loss_values, label="Train loss")
    plt.plot(epoch_count, test_loss_values, label="Test loss")
    plt.title(f"Training and test loss curves for model {model_name}")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.ylim(bottom=0) # Set bottom to 0, let top auto-adjust
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if output != "":
        os.makedirs(output, exist_ok=True)
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = "".join([c if c.isalnum() else "_" for c in model_name.replace(" ", "_")[:30]])
        filename = os.path.join(output, f"{safe_model_name}_loss_plot_{time_str}.png")
        try:
            plt.savefig(filename)
            print(f"Saved loss plot to {filename}")
        except Exception as e:
            print(f"Error saving loss plot: {e}")
    
    if show:
        plt.show()
    plt.close()


def predVStrue(label_val_true: list,      
               label_val_pred: list,      
               label_train_true: list,    
               label_train_pred: list,    
               model_name: str,
               output: str = "",
               show: bool = True) -> None:
    plt.figure(figsize=(8, 6)) 

    plotted_training = False
    if label_train_true and label_train_pred: 
        try:
            train_true_concat = np.concatenate(label_train_true)
            train_pred_concat = np.concatenate(label_train_pred)
            if train_true_concat.size > 0 and train_pred_concat.size > 0:
                plt.scatter(train_true_concat, train_pred_concat, label='Training Data', alpha=0.5, s=10, c='blue')
                plotted_training = True
        except ValueError as e:
            print(f"Warning: Could not process training data for predVStrue plot: {e}")

    plotted_validation = False
    if label_val_true and label_val_pred: 
        try:
            val_true_concat = np.concatenate(label_val_true)
            val_pred_concat = np.concatenate(label_val_pred)
            if val_true_concat.size > 0 and val_pred_concat.size > 0:
                plt.scatter(val_true_concat, val_pred_concat, label='Validation Data', alpha=0.5, s=10, c='orange')
                plotted_validation = True
        except ValueError as e:
            print(f"Warning: Could not process validation data for predVStrue plot: {e}")

    if not plotted_training and not plotted_validation:
        print("Warning: No data provided to plot in predVStrue. Skipping plot generation.")
        plt.close() 
        return

    all_true_for_line = []
    if plotted_training: all_true_for_line.append(train_true_concat)
    if plotted_validation: all_true_for_line.append(val_true_concat)
    
    if all_true_for_line: 
        combined_true_for_line = np.concatenate(all_true_for_line)
        if combined_true_for_line.size > 0:
            min_val = np.min(combined_true_for_line)
            max_val = np.max(combined_true_for_line)
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal', lw=1.5)

    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'Predictions vs True Values ({model_name})', fontsize=14)
    if plotted_training or plotted_validation: 
        plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if output != "" and isinstance(output, str):
        os.makedirs(output, exist_ok=True)
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
        safe_model_name = "".join([c if c.isalnum() else "_" for c in model_name.replace(" ", "_")[:30]])
        filename = os.path.join(output, f"{safe_model_name}_predVStrue_{time_str}.png")
        try:
            plt.savefig(filename)
            print(f"Saved predVStrue plot to {filename}")
        except Exception as e:
            print(f"Error saving predVStrue plot: {e}")
    
    if show:
        plt.show()
    plt.close()

def cuda_status(device : torch.device) -> None:
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

def plot_accuracy_vs_time(classical_times, classical_accuracies, gat_times, gat_accuracies, 
                          time_label='Inference Time (s)', 
                          accuracy_label='FE Solution Error (e.g., L2 Norm)', 
                          title='Accuracy vs. Inference Time Comparison',
                          output: str = "", show: bool = True, **kwargs):
    """
    Generates a scatter plot comparing the accuracy (error) vs. time
    of two methods. Lower and to the left is better.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(classical_times, classical_accuracies, label='Classical r-adaptivity', marker='o', s=80, alpha=0.7)
    plt.scatter(gat_times, gat_accuracies, label='GAT-based Optimization', marker='x', s=80, alpha=0.7)
    plt.xlabel(time_label)
    plt.ylabel(accuracy_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Use a log scale for time if the values vary widely
    if np.max(classical_times) / np.min(classical_times) > 10 or np.max(gat_times) / np.min(gat_times) > 10:
        plt.xscale('log')
    plt.tight_layout()

    if output != "" and isinstance(output, str):
        os.makedirs(output, exist_ok=True)
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title_part = "".join([c if c.isalnum() else "_" for c in title.replace(" ", "_")[:40]])
        filename = os.path.join(output, f"{safe_title_part}_acc_vs_time_{time_str}.png")
        try:
            plt.savefig(filename)
            print(f"Saved accuracy vs time plot to {filename}")
        except Exception as e:
            print(f"Error saving accuracy vs time plot: {e}")
    
    if show:
        plt.show()
    plt.close()

def plot_accuracy_vs_cost(classical_costs, classical_accuracies, gat_costs, gat_accuracies, 
                          cost_label='Computational Cost (e.g., Nodes)', 
                          accuracy_label='FE Solution Error (e.g., L2 Norm)', 
                          title='Accuracy vs. Computational Cost Comparison',
                          output: str = "", show: bool = True, **kwargs): # Added output, show, **kwargs
    plt.figure(figsize=(8, 6))
    plt.scatter(classical_costs, classical_accuracies, label='Classical r-adaptivity', marker='o', s=80, alpha=0.7)
    plt.scatter(gat_costs, gat_accuracies, label='GAT-based Optimization', marker='x', s=80, alpha=0.7)
    plt.xlabel(cost_label)
    plt.ylabel(accuracy_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if output != "" and isinstance(output, str):
        os.makedirs(output, exist_ok=True)
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title_part = "".join([c if c.isalnum() else "_" for c in title.replace(" ", "_")[:40]])
        filename = os.path.join(output, f"{safe_title_part}_acc_vs_cost_{time_str}.png")
        try:
            plt.savefig(filename)
            print(f"Saved accuracy vs cost plot to {filename}")
        except Exception as e:
            print(f"Error saving accuracy vs cost plot: {e}")
    
    if show:
        plt.show()
    plt.close()


def plot_time_comparison(classical_times, gat_times,
                         time_label='Mesh Optimization Time (s)',
                         title='Mesh Optimization Step Time Comparison',
                         use_box_plot=False,
                         output: str = "", 
                         show: bool = True, **kwargs): # Added **kwargs
    plt.figure(figsize=(7, 6)) 
    methods = ["Classical Method", "GAT Method"]

    if use_box_plot:
        c_times_plot = [classical_times] if np.isscalar(classical_times) else list(classical_times)
        g_times_plot = [gat_times] if np.isscalar(gat_times) else list(gat_times)
        if not c_times_plot or not g_times_plot: # Handle empty lists
            print("Warning: Empty data for box plot. Skipping.")
            plt.close()
            return

        data_to_plot = [c_times_plot, g_times_plot]
        bp = plt.boxplot(data_to_plot, labels=methods, patch_artist=True, vert=True, widths=0.6)
        colors = ['skyblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)
    else:  
        avg_classical_time = np.mean(classical_times) if classical_times else np.nan
        avg_gat_time = np.mean(gat_times) if gat_times else np.nan
        times_to_plot = [avg_classical_time, avg_gat_time]

        if np.isnan(avg_classical_time) or np.isnan(avg_gat_time):
            print("Warning: NaN times for bar chart. Skipping.")
            plt.close()
            return

        bars = plt.bar(methods, times_to_plot, color=['skyblue', 'lightcoral'], width=0.6)
        max_h = max(times_to_plot, default=1) if times_to_plot else 1
        for bar_item in bars:
            yval = bar_item.get_height()
            offset = 0.01 * max_h 
            plt.text(bar_item.get_x() + bar_item.get_width()/2.0, yval + offset,
                     f'{yval:.2e}', ha='center', va='bottom', fontsize=9)

    plt.ylabel(time_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if output != "" and isinstance(output, str):
        os.makedirs(output, exist_ok=True) 
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
        safe_title_part = "".join([c if c.isalnum() else "_" for c in title.replace(" ", "_")[:40]])
        filename = os.path.join(output, f"{safe_title_part}_time_comp_{time_str}.png")
        try:
            plt.savefig(filename)
            print(f"Saved time comparison plot to {filename}")
        except Exception as e:
            print(f"Error saving time comparison plot: {e}")
    
    if show:
        plt.show()
    plt.close() 

# MODIFIED plot_convergence
def plot_convergence(classical_dofs, classical_errors, gat_dofs, gat_errors, 
                     dof_label='Degrees of Freedom (DoF)', 
                     error_label='FE Solution Error (e.g., L2 Norm)', 
                     title='Convergence Plot',
                     output: str = "",      # Added output parameter
                     show: bool = True,     # Added show parameter
                     **kwargs):             # Added **kwargs for flexibility
    """
    Generates a line plot comparing the convergence (error vs. DoF/nodes)
    of two methods.
    """
    plt.figure(figsize=(8, 6))

    # Ensure data is list or np.array and not empty before sorting
    if not classical_dofs or not classical_errors or not gat_dofs or not gat_errors:
        print("Warning: Empty data provided for convergence plot. Skipping.")
        plt.close()
        return

    try:
        classical_dofs_arr = np.array(classical_dofs)
        classical_errors_arr = np.array(classical_errors)
        gat_dofs_arr = np.array(gat_dofs)
        gat_errors_arr = np.array(gat_errors)

        classical_sorted_indices = np.argsort(classical_dofs_arr)
        gat_sorted_indices = np.argsort(gat_dofs_arr)

        plt.plot(classical_dofs_arr[classical_sorted_indices], classical_errors_arr[classical_sorted_indices],
                 label='Classical r-adaptivity', marker='o', linestyle='-')
        plt.plot(gat_dofs_arr[gat_sorted_indices], gat_errors_arr[gat_sorted_indices],
                 label='GAT-based Optimization', marker='x', linestyle='-')
    except Exception as e_plot:
        print(f"Error during plotting convergence data: {e_plot}")
        plt.close()
        return

    plt.xlabel(dof_label)
    plt.ylabel(error_label)
    plt.xscale('log') 
    plt.yscale('log') 
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if output != "" and isinstance(output, str):
        os.makedirs(output, exist_ok=True)
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title_part = "".join([c if c.isalnum() else "_" for c in title.replace(" ", "_")[:40]])
        filename = os.path.join(output, f"{safe_title_part}_convergence_{time_str}.png")
        try:
            plt.savefig(filename)
            print(f"Saved convergence plot to {filename}")
        except Exception as e_save:
            print(f"Error saving convergence plot: {e_save}")

    if show:
        plt.show()
    plt.close()


def plot_gat_training_loss(epochs, training_loss, validation_loss=None, 
                           loss_label='Loss', title='GAT Model Training Progress',
                           output: str = "", show: bool = True, **kwargs): # Added output, show, **kwargs
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, training_loss, label='Training Loss', linestyle='-', color='blue')
    if validation_loss is not None:
        plt.plot(epochs, validation_loss, label='Validation Loss', linestyle='--', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel(loss_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    if output != "" and isinstance(output, str):
        os.makedirs(output, exist_ok=True)
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title_part = "".join([c if c.isalnum() else "_" for c in title.replace(" ", "_")[:40]])
        filename = os.path.join(output, f"{safe_title_part}_gat_train_loss_{time_str}.png")
        try:
            plt.savefig(filename)
            print(f"Saved GAT training loss plot to {filename}")
        except Exception as e:
            print(f"Error saving GAT training loss plot: {e}")

    if show:
        plt.show()
    plt.close()


def weisfeiler_lehman_test(G, iterations=3):
    node_labels = {node: str(G.degree(node)) for node in G.nodes()}
    label_history = [] 
    label_history.append(Counter(node_labels.values()))
    print(f"Iteration 0 (Initial): {len(label_history[0])} unique labels")

    for k in range(iterations):
        new_labels = {}
        aggregate_signatures = {} 
        for node in G.nodes():
            neighbor_labels = sorted([node_labels[neighbor] for neighbor in G.neighbors(node)])
            signature = (node_labels[node], tuple(neighbor_labels))
            aggregate_signatures[node] = signature
        
        unique_signatures = sorted(list(set(aggregate_signatures.values())))
        signature_to_compressed_label = {}
        for i, sig in enumerate(unique_signatures):
            hash_object = hashlib.sha256(str(sig).encode())
            compressed_label = hash_object.hexdigest()[:8]
            signature_to_compressed_label[sig] = compressed_label
        
        for node in G.nodes():
            new_labels[node] = signature_to_compressed_label[aggregate_signatures[node]]
        
        node_labels = new_labels
        current_histogram = Counter(node_labels.values())
        label_history.append(current_histogram)
        print(f"Iteration {k+1}: {len(current_histogram)} unique labels")
        if k > 0 and current_histogram == label_history[-2]:
            print(f"Converged after {k+1} iterations.")
            break
    return label_history

def plot_wl_histograms(label_history, graph_name="Graph", 
                       output: str = "", show: bool = True, **kwargs): # Added output, show, **kwargs
    num_iterations = len(label_history)
    if num_iterations == 0:
        print("Warning: Empty label history for WL plot. Skipping.")
        return

    fig, axes = plt.subplots(1, num_iterations, figsize=(5 * num_iterations, 4), sharey=True)
    if num_iterations == 1: 
        axes = [axes]

    fig.suptitle(f'Weisfeiler-Lehman Label Distribution for {graph_name}', fontsize=16)
    unique_labels_across_iterations = set()
    for hist in label_history:
        unique_labels_across_iterations.update(hist.keys())
    sorted_unique_labels = sorted(list(unique_labels_across_iterations))
    num_unique_labels = len(sorted_unique_labels)

    for i, hist in enumerate(label_history):
        ax = axes[i]
        counts = [hist.get(label, 0) for label in sorted_unique_labels]
        indices = np.arange(num_unique_labels) 
        ax.bar(indices, counts, tick_label=sorted_unique_labels)
        ax.set_title(f'Iteration {i}')
        ax.set_xlabel('Node Label')
        if i == 0:
            ax.set_ylabel('Number of Nodes')
        ax.tick_params(axis='x', rotation=90) 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    if output != "" and isinstance(output, str):
        os.makedirs(output, exist_ok=True)
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_graph_name = "".join([c if c.isalnum() else "_" for c in graph_name.replace(" ", "_")[:30]])
        filename = os.path.join(output, f"{safe_graph_name}_wl_histograms_{time_str}.png")
        try:
            plt.savefig(filename)
            print(f"Saved WL histogram plot to {filename}")
        except Exception as e:
            print(f"Error saving WL histogram plot: {e}")
    
    if show:
        plt.show()
    plt.close()
