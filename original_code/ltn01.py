import torch
import torch.nn as nn
import ltn
import importlib.metadata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from z3 import Solver, Bool, And, Not, sat

# Device selection: use GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Print library versions
print("torch version:", torch.__version__)
print("ltntorch version:", importlib.metadata.version("LTNtorch"))
try:
    print("z3-solver version:", importlib.metadata.version("z3-solver"))
except importlib.metadata.PackageNotFoundError:
    print("z3-solver not found. Please install it using: pip install z3-solver")
print("="*20)


# 1. Define Constants as Learnable Embeddings
EMBEDDING_DIM = 5
MAX_NUM_OF_ITERATIONS = 1000 # Increased epochs for more complex constraints
MAX_NUM_OF_PRINTABLE_LINES = 10 # Limit for printing large tensors in the console output

# Define a threshold to convert fuzzy truth values to boolean
TRUTH_THRESHOLD = 0.95

# Constants for the analogy
king = ltn.Constant(torch.randn(EMBEDDING_DIM, device=device), trainable=True)
man = ltn.Constant(torch.randn(EMBEDDING_DIM, device=device), trainable=True)
woman = ltn.Constant(torch.randn(EMBEDDING_DIM, device=device), trainable=True)
queen = ltn.Constant(torch.randn(EMBEDDING_DIM, device=device), trainable=True)

# Constants for the synonym relationships
car = ltn.Constant(torch.randn(EMBEDDING_DIM, device=device), trainable=True)
automobile = ltn.Constant(torch.randn(EMBEDDING_DIM, device=device), trainable=True)
monarch = ltn.Constant(torch.randn(EMBEDDING_DIM, device=device), trainable=True)

# Constants for the antonym relationship
hot = ltn.Constant(torch.randn(EMBEDDING_DIM, device=device), trainable=True)
cold = ltn.Constant(torch.randn(EMBEDDING_DIM, device=device), trainable=True)

# Constants for the homonym relationship
bank_river = ltn.Constant(torch.randn(EMBEDDING_DIM, device=device), trainable=True)
bank_financial = ltn.Constant(torch.randn(EMBEDDING_DIM, device=device), trainable=True)
water = ltn.Constant(torch.randn(EMBEDDING_DIM, device=device), trainable=True)
money = ltn.Constant(torch.randn(EMBEDDING_DIM, device=device), trainable=True)


# 2. Define Functions and Predicates
class Addition(nn.Module):
    def forward(self, a, b):
        return a + b

class Subtraction(nn.Module):
    def forward(self, a, b):
        return a - b

class Similarity(nn.Module):
    def forward(self, a, b):
        distance_sq = torch.sum(torch.square(a - b))
        return torch.exp(-distance_sq)

class Dissimilarity(nn.Module):
    def forward(self, a, b):
        distance_sq = torch.sum(torch.square(a - b))
        # Returns a high value for large distances, low for small.
        # We use a tanh to keep the value bounded and smooth.
        return torch.tanh(distance_sq * 0.1)

add = ltn.Function(Addition())
subtract = ltn.Function(Subtraction())
isSimilar = ltn.Predicate(Similarity())
isOpposite = ltn.Predicate(Dissimilarity())

# 3. State the Logical Axioms (Soft Constraints for LTN)
# Axiom 1: The classic gender analogy
analogy_axiom = isSimilar(add(subtract(king, man), woman), queen)

# Axiom 2: The synonym relationship for car/automobile
synonym_axiom_1 = isSimilar(car, automobile)

# Axiom 3: The synonym relationship for king/monarch
synonym_axiom_2 = isSimilar(king, monarch)

# Axiom 4: The antonym relationship for hot/cold
antonym_axiom = isOpposite(hot, cold)

# Axioms 5 & 6: Disambiguating homonyms for "bank"
homonym_axiom_1 = isSimilar(bank_river, water)
homonym_axiom_2 = isSimilar(bank_financial, money)
# Axiom 7: Ensure the two meanings of bank are distinct
homonym_axiom_3 = isOpposite(bank_river, bank_financial)


# 4. Learn the Embeddings
learnable_params = [
    king.value, man.value, woman.value, queen.value,
    car.value, automobile.value, monarch.value,
    hot.value, cold.value,
    bank_river.value, bank_financial.value, water.value, money.value
]
optimizer = torch.optim.Adam(learnable_params, lr=0.1)

SatAgg = ltn.fuzzy_ops.AggregPMeanError(p=2) 

# Training loop
print("Starting LTN training...")
for i in range(MAX_NUM_OF_ITERATIONS):
    optimizer.zero_grad()
    
    # Recompute axioms each iteration for a fresh computation graph
    analogy_axiom = isSimilar(add(subtract(king, man), woman), queen)
    synonym_axiom_1 = isSimilar(car, automobile)
    synonym_axiom_2 = isSimilar(king, monarch)
    antonym_axiom = isOpposite(hot, cold)
    homonym_axiom_1 = isSimilar(bank_river, water)
    homonym_axiom_2 = isSimilar(bank_financial, money)
    homonym_axiom_3 = isOpposite(bank_river, bank_financial)
    
    # The knowledge base now contains all our axioms
    knowledge_base = torch.stack([
        analogy_axiom.value, 
        synonym_axiom_1.value, 
        synonym_axiom_2.value,
        antonym_axiom.value,
        homonym_axiom_1.value,
        homonym_axiom_2.value,
        homonym_axiom_3.value
    ])
    
    sat = SatAgg(knowledge_base)
    loss = 1.0 - sat
    loss.backward()
    optimizer.step()

    if i % (MAX_NUM_OF_ITERATIONS // MAX_NUM_OF_PRINTABLE_LINES) == 0:
        print(f"Epoch {i}, Loss: {loss.item():.4f}, Satisfiability: {sat.item():.4f}")
    
    if sat.item() > 0.99:
        print(f"Early stopping at epoch {i}, Satisfiability: {sat.item():.4f}")
        break

print("-" * 20)
print("LTN training finished.")

# Recompute final axiom values for printing
analogy_axiom = isSimilar(add(subtract(king, man), woman), queen)
synonym_axiom_1 = isSimilar(car, automobile)
synonym_axiom_2 = isSimilar(king, monarch)
antonym_axiom = isOpposite(hot, cold)
homonym_axiom_1 = isSimilar(bank_river, water)
homonym_axiom_2 = isSimilar(bank_financial, money)
homonym_axiom_3 = isOpposite(bank_river, bank_financial)

print("\nFinal satisfiability of analogy_axiom:", analogy_axiom.value.item())
print("Final satisfiability of synonym_axiom_1 (car/auto):", synonym_axiom_1.value.item())
print("Final satisfiability of synonym_axiom_2 (king/monarch):", synonym_axiom_2.value.item())
print("Final satisfiability of antonym_axiom (hot/cold):", antonym_axiom.value.item())
print("Final satisfiability of homonym_axiom_1 (bank_river/water):", homonym_axiom_1.value.item())
print("Final satisfiability of homonym_axiom_2 (bank_financial/money):", homonym_axiom_2.value.item())
print("Final satisfiability of homonym_axiom_3 (bank vs bank):", homonym_axiom_3.value.item())


# 5. Verification
print("\n--- Antonym Verification ---")
print("Learned 'hot' embedding: ", hot.value.detach().cpu().numpy())
print("Learned 'cold' embedding:", cold.value.detach().cpu().numpy())
antonym_distance = torch.sum(torch.square(hot.value - cold.value)).item()
print(f"Squared distance between hot and cold: {antonym_distance:.4f}")

print("\n--- Homonym Verification ---")
print("Learned 'bank_river' embedding:    ", bank_river.value.detach().cpu().numpy())
print("Learned 'bank_financial' embedding:", bank_financial.value.detach().cpu().numpy())
homonym_distance = torch.sum(torch.square(bank_river.value - bank_financial.value)).item()
print(f"Squared distance between the two 'bank' meanings: {homonym_distance:.4f}")

# 6. Hard Constraint Verification with SMT Solver (Z3)
print("\n--- SMT Solver Verification ---")


# Create boolean variables for the SMT solver
# These represent the high-confidence beliefs learned by the LTN
p_isSimilar_king_monarch = Bool("isSimilar_king_monarch")
p_isOpposite_hot_cold = Bool("isOpposite_hot_cold")
p_analogy_holds = Bool("analogy_holds")

# Create a solver instance
s = Solver()

# Translate the LTN's learned beliefs into boolean facts for the solver
s.add(p_analogy_holds == (analogy_axiom.value.item() > TRUTH_THRESHOLD))
s.add(p_isSimilar_king_monarch == (synonym_axiom_2.value.item() > TRUTH_THRESHOLD))
s.add(p_isOpposite_hot_cold == (antonym_axiom.value.item() > TRUTH_THRESHOLD))

print(f"Adding learned facts to SMT solver (threshold={TRUTH_THRESHOLD}):")
print(f"  - Analogy holds: {analogy_axiom.value.item() > TRUTH_THRESHOLD}")
print(f"  - King is similar to Monarch: {synonym_axiom_2.value.item() > TRUTH_THRESHOLD}")
print(f"  - Hot is opposite to Cold: {antonym_axiom.value.item() > TRUTH_THRESHOLD}")

# Define a new, HARD constraint that was not used in training.
# For example: "The analogy must hold AND the king/monarch synonym must be true."
hard_constraint = And(p_analogy_holds, p_isSimilar_king_monarch)
print(f"\nAdding hard constraint to solver: And(analogy_holds, isSimilar_king_monarch)")
s.add(hard_constraint)

# Check for satisfiability
result = s.check()

if result == sat:
    print("\nResult: SATISFIABLE")
    print("The learned embeddings are logically consistent with the hard constraint.")
    print("Model found by solver:")
    print(s.model())
else:
    print("\nResult: UNSATISFIABLE")
    print("The learned embeddings contradict the hard constraint.")

# Example of a contradictory hard constraint
print("\n--- Testing a Contradictory Hard Constraint ---")
s_contra = Solver()
s_contra.add(p_isOpposite_hot_cold == (antonym_axiom.value.item() > TRUTH_THRESHOLD))
# Add a hard constraint that is likely false: "hot and cold must NOT be opposites"
contradictory_constraint = Not(p_isOpposite_hot_cold)
print(f"Adding learned fact: Hot is opposite to Cold -> {antonym_axiom.value.item() > TRUTH_THRESHOLD}")
print("Adding hard constraint: Not(isOpposite_hot_cold)")
s_contra.add(contradictory_constraint)

result_contra = s_contra.check()
if result_contra == sat:
    print("\nResult: SATISFIABLE (This would indicate an issue)")
else:
    print("\nResult: UNSATISFIABLE")
    print("As expected, the learned knowledge contradicts this hard constraint.")


# 7. Dimensionality Analysis Report (PCA)
print("\n--- PCA Report ---")

# Gather all learned embeddings and their labels
labels = [
    'king', 'man', 'woman', 'queen', 'car', 'automobile', 'monarch',
    'hot', 'cold', 'bank_river', 'bank_financial', 'water', 'money'
]
embeddings = [
    king.value, man.value, woman.value, queen.value,
    car.value, automobile.value, monarch.value,
    hot.value, cold.value,
    bank_river.value, bank_financial.value, water.value, money.value
]

# Convert to a numpy array for sklearn
embeddings_np = np.array([e.detach().cpu().numpy() for e in embeddings])

# Perform PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_np)

# Create a scatter plot
plt.figure(figsize=(12, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

# Add labels to the points
for i, label in enumerate(labels):
    plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), textcoords="offset points", xytext=(5,5), ha='center')

plt.title('PCA of Learned Concept Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Print explained variance
print(f"\nExplained variance by component: {pca.explained_variance_ratio_}")
print(f"Total variance explained by 2 components: {np.sum(pca.explained_variance_ratio_):.2f}")
