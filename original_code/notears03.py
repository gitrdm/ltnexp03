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
    # The ground truth is a simple chain: 0 -> 1 -> 2.
    print("\nGenerating data with a known causal structure: 0 -> 1 -> 2")
    
    noise = torch.randn(num_samples, num_vars)
    data = torch.zeros(num_samples, num_vars)
    
    data[:, 0] = noise[:, 0]
    data[:, 1] = 0.8 * data[:, 0] + noise[:, 1]
    data[:, 2] = 0.8 * data[:, 1] + noise[:, 2]
    data[:, 3] = noise[:, 3]
    data[:, 4] = noise[:, 4]

    # Standardize the data
    data = (data - data.mean(axis=0)) / data.std(axis=0)

    # The "model" is just the learnable adjacency matrix W.
    W = torch.nn.Parameter(torch.zeros(num_vars, num_vars)) # Start with zeros
    
    # --- FIX: Implement the Augmented Lagrangian optimization method ---
    # This is a more robust way to handle the hard acyclicity constraint.
    lambda_l1 = 0.1
    rho = 1.0  # Penalty parameter
    alpha = 0.0 # Dual variable for the constraint
    h_val = np.inf # Current value of the acyclicity constraint

    print("\nStarting training with Augmented Lagrangian method...")
    for i in range(10): # Outer loop for updating rho and alpha
        print(f"\nOuter Loop {i+1}/10: rho={rho:.2f}, alpha={alpha:.2f}")
        
        optimizer = optim.LBFGS([W], lr=0.1, max_iter=100)

        # Inner loop for optimizing W
        for _ in range(5):
            def closure():
                optimizer.zero_grad()
                X_recon = data @ W
                recon_loss = torch.sum((data - X_recon)**2)
                h_val_current = h_func(W)
                
                # Augmented Lagrangian loss
                lagrangian_loss = recon_loss + alpha * h_val_current + (rho / 2) * h_val_current**2
                l1_loss = lambda_l1 * torch.sum(torch.abs(W))
                total_loss = lagrangian_loss + l1_loss
                
                total_loss.backward()
                return total_loss
            
            optimizer.step(closure)

        # Update the penalty parameters after the inner optimization
        with torch.no_grad():
            h_val_new = h_func(W).item()
        
        if h_val_new > 0.25 * h_val:
            rho *= 10
        
        alpha += rho * h_val_new
        h_val = h_val_new

        # Stop if the graph is acyclic
        if h_val < 1e-8:
            print("Acyclicity constraint met. Stopping early.")
            break

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
