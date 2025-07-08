import torch
import torch.optim as optim
import numpy as np

# This script provides a simplified implementation of the linear NOTEARS algorithm.
# It is much simpler than DAG-GNN because it assumes linear relationships
# and does not require a VAE or GNN architecture.

# --- 1. The Differentiable Acyclicity Constraint ---
# This function is identical to the one used in the DAG-GNN example.
def h_func(W):
    """
    Calculates the acyclicity constraint.
    h(W) = tr(e^(W * W)) - d
    """
    d = W.shape[0]
    expm_W = torch.matrix_exp(W * W)
    h = torch.einsum('ii->', expm_W) - d
    return h

# --- 2. The Linear NOTEARS Training Process ---
def train_linear_notears():
    print("--- Training Linear NOTEARS ---")

    # Configuration
    num_vars = 5
    num_samples = 1000

    # Generate some dummy data
    # In a real scenario, this would be your dataset.
    data = torch.randn(num_samples, num_vars)

    # The "model" is just the learnable adjacency matrix W.
    # There is no complex neural network.
    W = torch.nn.Parameter(torch.rand(num_vars, num_vars))
    optimizer = optim.Adam([W], lr=0.01)

    print("Starting training...")
    for epoch in range(300):
        optimizer.zero_grad()

        # The core linear model assumption: X ≈ XW
        # We calculate the reconstructed data based on this linear relationship.
        X_recon = data @ W

        # --- Calculate the two loss components ---
        # 1. Reconstruction Loss (how well XW reconstructs X)
        recon_loss = torch.sum((data - X_recon)**2)

        # 2. Acyclicity Loss (the penalty for having cycles)
        acyclicity_loss = h_func(W)

        # Combine the losses
        lambda_acyclic = 1.0
        total_loss = recon_loss + lambda_acyclic * acyclicity_loss

        # Backpropagation and optimization
        total_loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}, "
                  f"Recon Loss: {recon_loss.item():.4f}, "
                  f"Acyclicity Loss: {acyclicity_loss.item():.4f}")

    print("\nTraining finished.")
    print("-" * 20)

    # The learned causal graph is the final state of W
    learned_W = W.detach().numpy()

    # We can threshold the matrix to get a binary adjacency matrix
    threshold = 0.3
    learned_graph = (np.abs(learned_W) > threshold).astype(int)

    print("Learned Adjacency Matrix (W):\n", learned_W.round(2))
    print("\nFinal Causal Graph (thresholded at 0.3):\n", learned_graph)

if __name__ == "__main__":
    train_linear_notears()
