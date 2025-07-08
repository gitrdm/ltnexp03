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
    # The matrix exponential is a key component of the NOTEARS constraint
    expm_W = torch.matrix_exp(W * W) 
    # The trace of the matrix exponential is used to check for cycles
    h = torch.einsum('ii->', expm_W) - d
    return h

# --- 2. The Linear NOTEARS Training Process ---
def train_linear_notears():
    print("--- Training Linear NOTEARS ---")

    # Configuration
    num_vars = 5
    num_samples = 1000

    # --- Generate Data with a Known Causal Structure ---
    # Instead of random data, we now create data with a ground truth causal graph.
    # The ground truth is a simple chain: 0 -> 1 -> 2.
    # Variables 3 and 4 are independent noise.
    print("\nGenerating data with a known causal structure: 0 -> 1 -> 2")
    
    # Create independent noise for each variable
    noise = torch.randn(num_samples, num_vars)
    
    # Create the data tensor
    data = torch.zeros(num_samples, num_vars)
    
    # Define the causal relationships
    data[:, 0] = noise[:, 0]
    data[:, 1] = 0.8 * data[:, 0] + noise[:, 1] # Var 1 is caused by Var 0
    data[:, 2] = 0.8 * data[:, 1] + noise[:, 2] # Var 2 is caused by Var 1
    data[:, 3] = noise[:, 3]                     # Var 3 is independent
    data[:, 4] = noise[:, 4]                     # Var 4 is independent


    # The "model" is just the learnable adjacency matrix W.
    # There is no complex neural network.
    W = torch.nn.Parameter(torch.rand(num_vars, num_vars))
    optimizer = optim.Adam([W], lr=0.01)

    print("\nStarting training...")
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
        # The lambda value weights the importance of the acyclicity constraint.
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

    # We must set the diagonal to zero before thresholding to remove self-loops
    np.fill_diagonal(learned_W, 0)
    
    # We can threshold the matrix to get a binary adjacency matrix
    threshold = 0.3
    learned_graph = (np.abs(learned_W) > threshold).astype(int)

    print("Learned Adjacency Matrix (W):\n", learned_W.round(2))
    print("\nFinal Causal Graph (thresholded at 0.3):\n", learned_graph)
    print("\nExpected Graph:\n[[0 1 0 0 0]\n [0 0 1 0 0]\n [0 0 0 0 0]\n [0 0 0 0 0]\n [0 0 0 0 0]]")


if __name__ == "__main__":
    train_linear_notears()
