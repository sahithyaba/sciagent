from pathlib import Path
import matplotlib.pyplot as plt

def plot_series(df, column, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], df[column])
    ax.set_title(column)
    ax.set_xlabel("Time")
    ax.set_ylabel(column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
