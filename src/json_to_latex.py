# json_to_latex_2.py
import json
import argparse
import os

def escape_latex(text):
    """Escapes special LaTeX characters in a string."""
    if not isinstance(text, str):
        text = str(text)
    return text.replace('_', r'\_').replace('%', r'\%').replace('#', r'\#').replace('&', r'\&')

def format_stat_value(value, precision=6, sci_threshold=1e-3):
    """Formats a number for the stats table, using scientific notation for small values."""
    if value is None:
        return "N/A"
    try:
        # Use scientific notation for small, non-zero numbers
        if 0 < abs(float(value)) < sci_threshold:
            return f"{value:.{precision-2}e}"
        return f"{value:.{precision}f}"
    except (ValueError, TypeError):
        return "N/A"

def format_ms_value(value_seconds, precision=2):
    """Formats a time value from seconds to milliseconds."""
    if value_seconds is None:
        return "N/A"
    try:
        value_ms = float(value_seconds) * 1000.0
        return f"{value_ms:.{precision}f}"
    except (ValueError, TypeError):
        return "N/A"

def generate_stats_table(data):
    """Generates the detailed statistics LaTeX table."""
    params = data.get("parameters", {})
    model_name = escape_latex(data.get("model_name", "N/A"))
    session_ts = escape_latex(data.get("session_timestamp", "N/A"))
    label_ts = session_ts.replace(r'\_', '')
    device = escape_latex(data.get("device", "N/A"))
    val_samples = params.get("num_validation_samples_benchmarked", "N/A")
    inf_runs = params.get("inference_runs_per_sample", "N/A")
    
    classical_stats = data.get("classical_r_adaptivity_times_seconds", {})
    gat_stats = data.get("gat_inference_times_seconds", {})

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{Time Benchmark for: {model_name} (Session: {session_ts}). Device: {device}. Validation Samples: {val_samples}. GAT inference avg over {inf_runs} runs/sample.}}",
        f"  \\label{{tab:benchmark_{label_ts}}}",
        r"  \begin{tabular}{lrr}",
        r"    \toprule",
        r"    Metric & Classical R-Adaptivity (s) & GAT Inference (s) \\",
        r"    \midrule",
        f"    Mean     & {format_stat_value(classical_stats.get('mean'))} & {format_stat_value(gat_stats.get('mean'))} \\\\",
        f"    Median   & {format_stat_value(classical_stats.get('median'))} & {format_stat_value(gat_stats.get('median'))} \\\\",
        f"    Std. Dev.& {format_stat_value(classical_stats.get('std_dev'))} & {format_stat_value(gat_stats.get('std_dev'))} \\\\",
        f"    Min      & {format_stat_value(classical_stats.get('min'))} & {format_stat_value(gat_stats.get('min'))} \\\\",
        f"    Max      & {format_stat_value(classical_stats.get('max'))} & {format_stat_value(gat_stats.get('max'))} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}"
    ]
    return "\n".join(lines)

def generate_comparison_table(data):
    """Generates the simple mean time comparison LaTeX table."""
    model_name = escape_latex(data.get("model_name", "N/A"))
    session_ts = escape_latex(data.get("session_timestamp", "N/A"))
    label_ts = session_ts.replace(r'\_', '')
    
    classical_mean_s = data.get("classical_r_adaptivity_times_seconds", {}).get("mean")
    gat_mean_s = data.get("gat_inference_times_seconds", {}).get("mean")

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{Mesh Adaption Time Comparison: {model_name} (Session: {session_ts}).}}",
        f"  \\label{{tab:time_comp_{label_ts}}}",
        r"  \begin{tabular}{lr}",
        r"    \toprule",
        r"    Method                         & Time (ms) \\",
        r"    \midrule",
        f"    Classical R-Adaptivity (Dummy) & {format_ms_value(classical_mean_s, precision=1)} \\\\",
        f"    GAT R-Adaptivity (Inference)   & {format_ms_value(gat_mean_s, precision=2)} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}"
    ]
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Convert a diagnostic JSON file to LaTeX tables.")
    parser.add_argument("json_file", help="Path to the input JSON diagnostic file.")
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: File not found at {args.json_file}")
        return

    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    stats_table = generate_stats_table(data)
    comparison_table = generate_comparison_table(data)

    print("\n--- LaTeX Code ---")
    print(stats_table)
    print("\n" + "%" + "-"*78 + "%\n")
    print(comparison_table)
    print("\n--- End LaTeX Code ---")
    print("\nNote: You might need to include \\usepackage{booktabs} in your LaTeX preamble.")

if __name__ == "__main__":
    main()
