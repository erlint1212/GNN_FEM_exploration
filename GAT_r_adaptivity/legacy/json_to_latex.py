# In json_to_latex.py

import json
import argparse
import os # Make sure os is imported

def escape_latex(text):
    """Escapes special LaTeX characters in a string."""
    if not isinstance(text, str):
        text = str(text)
    # Basic escaping, you might need to add more characters if they appear in your data
    return text.replace('_', r'\_').replace('%', r'\%').replace('#', r'\#').replace('&', r'\&')

def format_value(value, precision=6, scientific_threshold_abs=1e-4):
    """Formats a float value, using scientific notation for small absolute values."""
    if isinstance(value, (int, float)):
        if value == 0: # Avoid -0.00e+00 for zero
            return "0.00" 
        if 0 < abs(value) < scientific_threshold_abs or abs(value) >= 1e5 :
            return f"{value:.2e}"
        return f"{value:.{precision}f}"
    return escape_latex(str(value)) # Escape string fallback

def generate_latex_table(data):
    """
    Generates a LaTeX table string from the benchmark data.
    """
    model_name = escape_latex(data.get("model_name", "N/A"))
    session_timestamp = escape_latex(data.get("session_timestamp", "N/A"))
    device = escape_latex(data.get("device", "N/A"))
    
    params = data.get("parameters", {})
    num_val_samples = params.get("num_validation_samples_benchmarked", "N/A")
    
    classical_stats = data.get("classical_r_adaptivity_times_seconds", {})
    gat_stats = data.get("gat_inference_times_seconds", {})
    gat_runs_per_sample = gat_stats.get("runs_per_sample_for_avg", "N/A")

    label_ts = session_timestamp.replace(r'\_', '') # For LaTeX label, remove escape for underscore

    # Using a list of strings to build the table, then join
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{Time Benchmark for: {model_name} (Session: {session_timestamp}). " +
                 f"Device: {device}. Validation Samples: {num_val_samples}. " +
                 f"GAT inference avg over {gat_runs_per_sample} runs/sample.}}")
    lines.append(f"  \\label{{tab:benchmark_{label_ts}}}")
    lines.append(r"  \begin{tabular}{lrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Metric & Classical R-Adaptivity (s) & GAT Inference (s) \\")
    lines.append(r"    \midrule")
    lines.append(f"    Mean     & {format_value(classical_stats.get('mean'))} & {format_value(gat_stats.get('mean'))} \\\\")
    lines.append(f"    Median   & {format_value(classical_stats.get('median'))} & {format_value(gat_stats.get('median'))} \\\\")
    lines.append(f"    Std. Dev.& {format_value(classical_stats.get('std_dev'))} & {format_value(gat_stats.get('std_dev'))} \\\\")
    lines.append(f"    Min      & {format_value(classical_stats.get('min'))} & {format_value(gat_stats.get('min'))} \\\\")
    lines.append(f"    Max      & {format_value(classical_stats.get('max'))} & {format_value(gat_stats.get('max'))} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}") # No need for double braces if not an f-string line
    lines.append(r"\end{table}")   # No need for double braces if not an f-string line

    return "\n".join(lines)

# main() function remains the same as previously provided
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
