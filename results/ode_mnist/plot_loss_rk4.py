
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
        # "results/ode_mnist/float32_torchmpnode_euler_seed24_20250304_153340/float32_torchmpnode_euler_seed24_20250304_153340.csv",
        "results/ode_mnist/float32_torchmpnode_rk4_seed24_20250304_154631/float32_torchmpnode_rk4_seed24_20250304_154631.csv",
        # "results/ode_mnist/float16_torchmpnode_euler_seed24_20250304_153340/float16_torchmpnode_euler_seed24_20250304_153340.csv"
         "results/ode_mnist/float16_torchmpnode_rk4_seed24_20250304_154631/float16_torchmpnode_rk4_seed24_20250304_154631.csv"
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
    column_to_plot = "val acc"
    
    plot_experiments(filenames, column_to_plot, legends, line_styles, markers)