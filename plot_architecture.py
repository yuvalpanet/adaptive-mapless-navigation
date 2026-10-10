import matplotlib.pyplot as plt


def draw_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Block positions
    nodes = {
        "obs": (0.10, 0.5),
        "encoder": (0.32, 0.5),
        "latent": (0.53, 0.5),
        "predictor": (0.80, 0.82),
        "actor_critic": (0.80, 0.18)
    }

    # Block labels
    labels = {
        "obs": "Raw Observation $s_t$\n(51)",
        "encoder": "State Encoder\n(AvgL1Norm)",
        "latent": "Latent $z_t$\n(256)",
        "predictor": "Latent Predictor\n$\\hat z_{t+1} \\ldots \\hat z_{t+k}$\n(actions taken, k = 3)",
        "actor_critic": "Actor $\\pi(s, z)$\nCritic $Q(s, a, z, \\hat z)$"
    }

    # One colour per stage
    colors = {
        "obs": "#fbb4ae",  # light red: input
        "encoder": "#b3cde3",  # light blue: encoder
        "latent": "#ccebc5",  # light green: latent space
        "predictor": "#decbe4",  # light purple: predictor
        "actor_critic": "#fed9a6"  # orange: actor / critic
    }

    # Draw the blocks
    for key, pos in nodes.items():
        ax.text(pos[0], pos[1], labels[key], ha="center", va="center", fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.8", facecolor=colors[key], edgecolor="black", linewidth=1.5))

    def arrow(start, end, color='black', style='-', rad=0.0):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle='-|>', lw=2, linestyle=style,
                                    mutation_scale=18, connectionstyle=f"arc3,rad={rad}"))

    def note(x, y, text, color='black'):
        ax.text(x, y, text, ha='center', va='center', fontsize=9.5, color=color, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9))

    # Input -> encoder -> latent
    arrow((0.195, 0.5), (0.245, 0.5))
    arrow((0.395, 0.5), (0.46, 0.5))

    # Latent -> predictor and -> actor / critic
    arrow((0.56, 0.58), (0.68, 0.80), rad=-0.2)
    arrow((0.56, 0.42), (0.68, 0.20), rad=0.2)

    # The raw observation also feeds the actor and critic
    arrow((0.10, 0.41), (0.68, 0.14), rad=0.25)
    note(0.45, 0.05, "raw $s$ (and $a$ for the critic)")

    # Predictor -> critic: the predicted next latent is a critic input
    arrow((0.80, 0.69), (0.80, 0.30))
    note(0.80, 0.5, "$\\hat z_{t+1}$")

    # Dashed red arrow: the prediction error becomes the intrinsic reward in the critic's target
    arrow((0.92, 0.74), (0.92, 0.26), color='red', style='--')
    ax.text(0.945, 0.5, "Intrinsic reward\n(normalised\nprediction error)", ha='left', va='center',
            fontsize=9.5, color='red', fontweight='bold')

    # Save the figure
    plt.tight_layout()
    plt.savefig('architecture.png', dpi=300, bbox_inches='tight')
    print("Saved architecture.png successfully!")


if __name__ == "__main__":
    draw_architecture()
