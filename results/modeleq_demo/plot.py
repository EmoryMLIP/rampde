
import pandas as pd
import matplotlib.pyplot as plt
import os


csv_path = "modeleq_demo.csv"
df = pd.read_csv(csv_path)

output_dir = "./examples/results/modeleq_plots"
os.makedirs(output_dir, exist_ok=True)

# Plotting
for method in df['method'].unique():
    for dtype_f in df['dtype_f'].unique(): #
        subset = df[(df['method'] == method) & (df['dtype_f'] == dtype_f)]

        for error_metric in ['grad_A_relerr', 'grad_x_relerr', 'sol_relerr']:
            plt.figure(figsize=(10, 6))
            for dtype_y in sorted(subset['dtype_y'].unique()): #
                if dtype_y == 'torch.float32':
                    style = 's-'
                elif dtype_y == 'torch.float16':
                    style = 'o--'
                for package, color in [('torchdiffeq', 'blue'), ('torchmpnode', 'red')]:
                    data = subset[(subset['dtype_y'] == dtype_y) & (subset['package'] == package)]
                    label = f"{package}, y={dtype_y}"
                    plt.plot(data['n_steps'], data[error_metric], style, label=label, color=color)

            plt.xscale('log')
            plt.yscale('log')
            # plt.xlabel('number of timesteps', fontsize=18)
            # plt.ylabel('relative error', fontsize=18)
            # plt.title(f"{method}, f={dtype_f} ({error_metric})", fontsize=18)
            plt.legend(fontsize=16)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            filename = f"{method}_{dtype_f}_{error_metric}.png".replace("torch.", "")
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

