# json_to_latex.py

import json
import argparse
import os
import numpy as np # For np.mean if processing raw lists, though stats are pre-calculated

def escape_latex(text):
    """Escapes special LaTeX characters in a string."""
    if not isinstance(text, str):
        text = str(text)
    # Add more characters as needed
    return text.replace('_', r'\_').replace('%', r'\%').replace('#', r'\#').replace('&', r'\&')

def format_time_ms(value_seconds, precision=1):
    """Formats a time value from seconds to milliseconds with specified precision."""
    if isinstance(value_seconds, (int, float)):
        value_ms = value_seconds * 1000.0
        # For very small ms values, scientific notation might be better,
        # but paper uses fixed point for times.
        return f"{value_ms:.{precision}f}"
    return escape_latex(str(value_seconds)) # Fallback

def generate_latex_table(data):
    """
    Generates a LaTeX table string from the benchmark data, styled similarly
    to the tables in the G-Adaptivity paper (focusing on time).
    """
    model_name = escape_latex(data.get("model_name", "N/A"))
    session_timestamp = escape_latex(data.get("session_timestamp", "N/A"))
    device = escape_latex(data.get("device", "N/A"))
    
    params = data.get("parameters", {})
    num_val_samples = params.get("num_validation_samples_benchmarked", "N/A")
    
    classical_stats = data.get("classical_r_adaptivity_times_seconds", {})
    gat_stats = data.get("gat_inference_times_seconds", {})
    # The paper reports mean times, sometimes with std dev in text or figures.
    # We will report mean time here.
    classical_mean_time_s = classical_stats.get("mean")
    gat_mean_time_s = gat_stats.get("mean")

    # Sanitize timestamp for LaTeX label (basic version)
    label_ts = session_timestamp.replace(r'\_', '')


    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    caption_text = (f"Mesh Adaption Time Comparison: {model_name} (Session: {session_timestamp}). " +
                    f"Validation Samples: {num_val_samples}. Device: {device}. " +
                    f"Times are mean over validation set samples.")
    lines.append(f"  \\caption{{{caption_text}}}")
    lines.append(f"  \\label{{tab:time_comp_{label_ts}}}")
    lines.append(r"  \begin{tabular}{lr}") # Two columns: Method name (left-aligned), Time (right-aligned)
    lines.append(r"    \toprule")
    lines.append(r"    Method                        & Time (ms) \\")
    lines.append(r"    \midrule")
    lines.append(f"    Classical R-Adaptivity (Dummy) & {format_time_ms(classical_mean_time_s, precision=1)} \\\\")
    lines.append(f"    GAT R-Adaptivity (Inference)   & {format_time_ms(gat_mean_time_s, precision=2)} \\\\") # GAT is faster, more precision
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Convert a diagnostic JSON file to a LaTeX table.")
    parser.add_argument("json_file", help="Path to the input JSON diagnostic file.")
    
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: File not found at {args.json_file}")
        return

    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.json_file}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    latex_table = generate_latex_table(data)
    print("\n--- LaTeX Table Code ---")
    print(latex_table)
    print("\n--- End LaTeX Table Code ---")
    print("\nNote: You might need to include \\usepackage{booktabs} in your LaTeX preamble for \\toprule, \\midrule, \\bottomrule.")

if __name__ == "__main__":
    main()
