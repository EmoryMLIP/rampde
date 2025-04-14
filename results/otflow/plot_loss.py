
import pandas as pd
import matplotlib.pyplot as plt

def plot_experiments(filenames, column_name, legends, line_styles, markers):
    plt.figure(figsize=(10, 6))
    output_filename=f"{column_name}.png"
    
    font_size = 20
    for idx, fname in enumerate(filenames):
        # Read CSV file into a DataFrame.
        df = pd.read_csv(fname)
        # Plot the selected column vs iteration.
        plt.plot(df['iteration'], df[column_name], linewidth=4, label=legends[idx], linestyle=line_styles[idx])
    
    plt.xlabel("Iteration", fontsize=font_size)
    plt.ylabel(column_name, fontsize=font_size)
    # plt.title(f"{column_name} vs Iteration")
    plt.legend( fontsize=font_size)
    plt.grid(True)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.tight_layout()
    
    # Save the plot as a PNG file.
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")
    plt.show()

if __name__ == "__main__":
    # Dummy CSV filenames.
    filenames = [
        "results/otflow/float32_torchmpnode_rk4_seed24_20250304_102614/float32_torchmpnode_rk4_seed24_20250304_102614.csv",
        # "results/otflow/float32_torchdiffeq_rk4_seed24_20250304_102446/float32_torchdiffeq_rk4_seed24_20250304_102446.csv",
        "results/otflow/float16_torchmpnode_rk4_seed24_20250304_101643/float16_torchmpnode_rk4_seed24_20250304_101643.csv",
        # "results/otflow/float16_torchdiffeq_rk4_seed24_20250304_101643/float16_torchdiffeq_rk4_seed24_20250304_101643.csv"
    ]
    legends = [
        "fp32 - torchmpnode",
        # "fp32 - torchdiffeq",
        "fp16 - torchmpnode",
        # "fp16 - torchdiffeq"
    ]

    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^']


    
    # Specify the column to plot. Change this to any valid column from your CSV.
    column_to_plot = "val loss"
    
    plot_experiments(filenames, column_to_plot, legends, line_styles, markers)