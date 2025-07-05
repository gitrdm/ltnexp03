from z3 import RealVector, Solver, And, sat, unsat

# This script demonstrates how to use an SMT solver (Z3) to check for logical
# inconsistencies in a set of core axioms before using them in a 'soft'
# framework like LTNtorch.

def check_axiom_consistency():
    """
    Uses Z3 to check if a set of analogical axioms is logically consistent.
    """
    print("--- Starting Axiom Consistency Check with Z3 ---")

    # 1. Define Concepts as Abstract Vectors
    # We represent each concept as a vector of abstract Real numbers in Z3.
    # The dimension doesn't matter for this logical check, so we'll use 2.
    embedding_dim = 2
    king = RealVector('king', embedding_dim)
    man = RealVector('man', embedding_dim)
    woman = RealVector('woman', embedding_dim)
    queen = RealVector('queen', embedding_dim)

    # 2. Translate Axioms into Z3 Constraints
    # Axiom 1 (Correct): king - man + woman == queen
    # We need to assert this for each dimension of the vector.
    axiom1 = And([king[i] - man[i] + woman[i] == queen[i] for i in range(embedding_dim)])
    print("Asserting Axiom 1: king - man + woman == queen")

    # Axiom 2 (Incorrect): king - woman + man == queen
    axiom2 = And([king[i] - woman[i] + man[i] == queen[i] for i in range(embedding_dim)])
    print("Asserting Axiom 2: king - woman + man == queen")

    # 3. Add Common Sense Assumptions
    # This is the key step. The first two axioms only become contradictory
    # if we assume that 'man' and 'woman' are distinct concepts.
    # We state that their vectors cannot be identical.
    common_sense_axiom = And([man[i] != woman[i] for i in range(embedding_dim)])
    print("Asserting Common Sense: man != woman")


    # 4. Run the Solver to Check for Satisfiability
    solver = Solver()
    
    # Add all axioms to the solver
    solver.add(axiom1)
    solver.add(axiom2)
    solver.add(common_sense_axiom)

    print("\nChecking if all axioms can be true simultaneously...")
    result = solver.check()

    # 5. Report the Result
    if result == sat:
        print("\nResult: SATISFIABLE")
        print("This indicates an issue, as the axioms should be contradictory.")
        print("Model found by Z3 that satisfies all axioms:")
        print(solver.model())
    elif result == unsat:
        print("\nResult: UNSATISFIABLE")
        print("SUCCESS: The SMT solver correctly identified a logical contradiction in the axioms.")
        print("It is impossible for all these statements to be true at the same time.")
    else:
        print(f"\nSolver returned an unknown result: {result}")

if __name__ == "__main__":
    check_axiom_consistency()

