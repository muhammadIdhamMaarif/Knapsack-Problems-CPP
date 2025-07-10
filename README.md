# 🧠 Knapsack Solver — All-in-One C++ Implementation

This repository provides a **comprehensive C++ implementation** of all major **Knapsack Problem variants** and their solving strategies, designed for advanced algorithm design and analysis. All source code is located in a **single file `knapsack.cpp`**, including the main function and all class implementations. A supporting `knapsack.hpp` contains base class declarations.

---

## 📁 File Structure

```
📦 knapsack-solver/
├── knapsack.cpp       # All algorithm implementations and main()
├── example.cpp        # The example usage of the class
└── knapsack.hpp       # Abstract class: KnapsackBase and derived headers
```

---

## 🔍 Supported Variants & Algorithms

| Knapsack Variant          | Algorithms Implemented                                                                 |
|---------------------------|------------------------------------------------------------------------------------------|
| **ZeroOneKnapsack**       | `dp`, `dp-1d`, `recursive-memo`, `brute-force`, `bitmask`, `meet-middle`, `branch-bound`, `fptas`, `greedy-local`, `drl` |
| **FractionalKnapsack**    | `greedy`, `sort-then-fill`, `brute-force`, `bitmask`                                   |
| **BoundedKnapsack**       | `brute-force`, `bitmask`, `recursive-memo`, `dp`, `binary-split-dp`, `branch-bound`, `greedy-local` |
| **UnboundedKnapsack**     | `brute-force`, `recursive-memo`, `dp`, `dp-1d`                                         |
| **MultiDimensionalKnapsack** | `brute-force`, `recursive-memo`, `dp`                                              |
| **MultiObjectiveKnapsack**   | `brute-force`, `dp`, `lexicographic`, `weighted-sum`                                |
| **MultipleKnapsack**      | `brute-force`, `greedy`, `dp-each-bag`                                                 |
| **QuadraticKnapsack**     | `brute-force`, `greedy`, `dp-approx`                                                   |
| **StochasticKnapsack**    | `monte-carlo`, `greedy-expected`, `expected-dp`                                        |
| **MultiChoiceKnapsack**   | `brute-force`, `greedy`, `dp`                                                          |
| **MetaheuristicSolvers**  | `simulated-annealing`, `ant-colony`, `pso`, `ilp`, `constraint-programming`           |

---

## ⚙️ Compilation & Execution

### 🔧 Build with g++

```bash
g++ -std=c++17 -O2 -o knapsack knapsack.cpp
```

### ▶️ Run Example

```bash
./knapsack
```

---

## 🧠 Algorithm Design & Analysis

### ✅ ZeroOne Knapsack

| Algorithm            | Time Complexity          | Space Complexity     | Notes                                                  |
|----------------------|--------------------------|-----------------------|--------------------------------------------------------|
| `dp`                 | O(nW)                    | O(nW)                 | Full DP table                                          |
| `dp-1d`              | O(nW)                    | O(W)                  | Memory optimized version of DP                        |
| `recursive-memo`     | O(nW)                    | O(nW)                 | Top-down memoized recursion                           |
| `brute-force`        | O(2ⁿ)                    | O(n)                  | Explores all subsets                                  |
| `brute-force-bitmask`| O(2ⁿ·n)                  | O(1)                  | Uses bitmasking for subset generation                 |
| `meet-in-the-middle` | O(2^(n/2)·log(2^(n/2)))  | O(2^(n/2))            | Splits problem, efficient when n ≤ 40                 |
| `branch-and-bound`   | ≤ O(2ⁿ)                  | Variable              | Prunes search space with bounding                     |
| `fptas`              | O(n² / ε)                | O(n)                  | Approximation scheme with error bound                 |
| `greedy-local`       | O(n log n)               | O(n)                  | Greedy based on value/weight, fast but not optimal    |
| `drl`                | -                        | -                     | Learns solution policy via reinforcement learning     |

### ✅ Fractional Knapsack

| Algorithm            | Time Complexity  | Space Complexity | Notes                                        |
|----------------------|------------------|------------------|----------------------------------------------|
| `greedy`             | O(n log n)       | O(1)             | Optimal due to greedy-choice property        |
| `sort-then-fill`     | O(n log n)       | O(n)             | Pre-sort then iterate                        |
| `brute-force`        | O(2ⁿ)            | O(n)             | Not practical, for completeness              |
| `brute-force-bitmask`| O(2ⁿ·n)          | O(1)             | Uses bitmask combinations                    |

### ✅ Bounded Knapsack

| Algorithm            | Time Complexity        | Space Complexity   | Notes                                           |
|----------------------|------------------------|---------------------|-------------------------------------------------|
| `brute-force`        | O(2ⁿ)                  | O(n)                | Inefficient but complete                        |
| `brute-force-bitmask`| O(2ⁿ·n)                | O(1)                | Bitmask generation for bounded items            |
| `recursive-memo`     | O(n·C)                 | O(n·C)              | Memoization with quantity consideration         |
| `dp`                 | O(n·C)                 | O(n·C)              | Classical bounded DP                            |
| `binary-split-dp`    | O(n log q · C)         | O(C)                | Splits items using binary encoding              |
| `branch-and-bound`   | ≤ O(2ⁿ)                | Variable            | Uses upper bounds to prune                      |
| `greedy-local`       | O(n log n)             | O(n)                | Fast but approximate                            |

### ✅ Unbounded Knapsack

| Algorithm            | Time Complexity | Space Complexity | Notes                                      |
|----------------------|------------------|------------------|--------------------------------------------|
| `brute-force`        | Exponential      | O(n)             | Inefficient                                |
| `recursive-memo`     | O(n·C)           | O(n·C)           | Memoized top-down solution                 |
| `dp`                 | O(n·C)           | O(n·C)           | Classic table-based approach               |
| `dp-1d`              | O(n·C)           | O(C)             | Space optimized DP                         |

### ✅ MultiDimensional Knapsack

| Algorithm            | Time Complexity        | Space Complexity      | Notes                                       |
|----------------------|------------------------|------------------------|---------------------------------------------|
| `brute-force`        | O(2ⁿ)                  | O(n)                   | Explores all subsets                        |
| `recursive-memo`     | O(n·C1·C2...)           | O(n·C1·C2...)          | With dimensional memoization                |
| `dp`                 | O(n·C1·C2...)           | O(n·C1·C2...)          | Multi-dimensional DP                        |

### ✅ MultiObjective Knapsack

| Algorithm            | Time Complexity   | Space Complexity     | Notes                                       |
|----------------------|-------------------|-----------------------|---------------------------------------------|
| `brute-force`        | O(2ⁿ)             | O(n)                  | All possible objective combinations         |
| `dp`                 | O(n·C)            | O(n·C)                | Tracks multi-objective states               |
| `lexicographic`      | O(n·C)            | O(n·C)                | Prioritizes objectives in lexicographic order|
| `weighted-sum`       | O(n·C)            | O(n·C)                | Uses scalar weights to combine objectives   |

### ✅ Multiple Knapsack

| Algorithm            | Time Complexity         | Space Complexity   | Notes                                        |
|----------------------|-------------------------|---------------------|----------------------------------------------|
| `brute-force`        | Exponential             | O(n)                | Checks all assignments                       |
| `greedy`             | O(n log n)              | O(n)                | Assigns by value density                     |
| `dp-each-bag`        | O(k·n·C)                | O(k·C)              | Runs DP separately for each of k bags        |

### ✅ Quadratic Knapsack

| Algorithm            | Time Complexity       | Space Complexity   | Notes                                         |
|----------------------|-----------------------|---------------------|-----------------------------------------------|
| `brute-force`        | O(2ⁿ + n²)            | O(n²)               | Includes quadratic interactions               |
| `greedy`             | O(n²)                | O(n)                | Uses pairwise profit combinations             |
| `dp-approx`          | O(n²/ε)               | O(n)                | Approximation scheme                          |

### ✅ Stochastic Knapsack

| Algorithm            | Time Complexity   | Space Complexity  | Notes                                         |
|----------------------|-------------------|--------------------|-----------------------------------------------|
| `monte-carlo`        | O(k·n)            | O(n)               | Random simulations, k = number of samples     |
| `greedy-expected`    | O(n log n)        | O(n)               | Sorts by expected value                       |
| `expected-dp`        | O(n·C)            | O(n·C)             | DP with expected value states                 |

### ✅ MultiChoice Knapsack

| Algorithm            | Time Complexity     | Space Complexity  | Notes                                           |
|----------------------|---------------------|--------------------|-------------------------------------------------|
| `brute-force`        | Exponential         | O(n)               | All choice combinations                        |
| `greedy`             | O(n log n)          | O(n)               | Best in class per group                        |
| `dp`                 | O(n·C)              | O(n·C)             | With group constraints                         |

### ✅ Metaheuristic Solvers

| Algorithm               | Time Complexity         | Space Complexity | Notes                                           |
|-------------------------|-------------------------|------------------|-------------------------------------------------|
| `simulated-annealing`   | O(k·n)                  | O(n)             | Probabilistic escape from local optima         |
| `ant-colony`            | O(k·n²)                 | O(n²)            | Path-based pheromone learning                  |
| `pso`                   | O(k·n)                  | O(n)             | Swarm optimization with global learning        |
| `ilp`                   | Solver-dependent        | High             | Uses solvers like CPLEX/Gurobi for exact result|
| `constraint-programming`| Solver-dependent        | High             | Declarative constraint satisfaction            |


---

## 📊 Benchmarking

- Tracks execution time.
- Logs results automatically.
- Can be customized to benchmark only selected knapsack problem
- Editable benchmark variable
- Progress bar visually

### ✅ How To Benchmark
1. Build and run the file
   ```bash
    g++ -std=c++17 -O2 -o knapsack knapsack.cpp
   ./knapsack
   ```
3. You'll be greeted with few options
   ```bash
    Knapsack Problem Solver
    Choose input method:
    1. Random generation
    2. Manual input
    3. Save to file
    4. Save all
    5. Exit
    Enter Choice (1 or 5): 
   ```
5. Choose ***Save All*** to benchmark all algorithm
6. Choose ***Save to File*** to choose which knapsack problem type to be test.
7. If you choose *Save to File*, it will provided all of the knapsack type
   ```bash
    Select knapsack type:
    1. 0/1 Knapsack
    2. Fractional Knapsack
    3. Bounded Knapsack
    4. Unbounded Knapsack
    5. Multi-Dimensional Knapsack
    6. Multi-Objective Knapsack
    7. Multiple Knapsack
    8. Quadratic Knapsack
    9. Stochastic Knapsack
    10. Multiple-Choice Knapsack
    11. Metaheuristic Approaches
    Enter your choice (1-11): 
   ```
9. All file will be saved to their respective knapsack type. It will save into `output/filename`.
10. You can edit the filename saved in `output` by editing the variable directly in code
    ```cpp
    #define OUTPUT_ZERO_ONE_KNAPSACK            "zero_one_knapsack.csv"
    #define OUTPUT_FRACTIONAL_KNAPSACK          "fractional_knapsack.csv"
    #define OUTPUT_BOUNDED_KNAPSACK             "bounded_knapsack.csv"
    #define OUTPUT_UNBOUNDED_KNAPSACK           "unbounded_knapsack.csv"
    #define OUTPUT_MULTI_DIMENSIONAL_KNAPSACK   "multi_dimensional_knapsack.csv"
    #define OUTPUT_MULTI_OBJECTIVE_KNAPSACK     "multi_objective_knapsack.csv"
    #define OUTPUT_MULTIPLE_KNAPSACK            "multiple_knapsack.csv"
    #define OUTPUT_QUADRATIC_KNAPSACK           "quadratic_knapsack.csv"
    #define OUTPUT_STOCHASTIC_KNAPSACK          "stochastic_knapsack.csv"
    #define OUTPUT_MULTI_CHOICE_KNAPSACK        "multi_choice_knapsack.csv"
    #define OUTPUT_METAHEURISTIC_KNAPSACK       "metaheuristic_knapsack.csv"
    ```
11. All benchmark will generate the data randomly ranging from 1 - some number
12. You can edit the benchmark range by editing directly in code
    ```cpp
    // Variabel maximum ti test
    #define MAX_NUMBER_ITEMS_TESTED     20
    #define MAX_ITEMS_VALUE_TESTED      1000     
    #define MAX_ITEMS_WEIGHT_TESTED     1000    
    #define MAX_QUANTITY_TESTED         5           
    #define MAX_CAPACITY_TESTED         1000        

    // Increment per iteration
    #define STEP_NUMBER_ITEMS           1
    #define STEP_ITEMS_VALUE            100
    #define STEP_ITEMS_WEIGHT           100
    #define STEP_QUANTITY               1
    #define STEP_CAPACITY               100
    ```

13. If your benchmark is slow, you can disable brute force
    ```cpp
    #define DISABLE_BRUTE_FORCE         true
    // you can change it to "true" or "false"
    ```

---

## 📌 Research and Academic Use

Designed for:
- Algorithm Design & Analysis courses
- Benchmarking of exact vs heuristic methods
- Approximation vs optimality studies
- Research on complex variants (multi-objective, stochastic, etc.)

---

## 🔬 Complexity Analysis Approach

Complexity metrics were derived via:
- Recurrence relations
- DP table size estimation
- Combinatorial state space evaluation
- Branching factors (for BnB, recursion)
- Empirical testing using benchmarks

---

## 📄 License

MIT License — use, modify, and distribute freely for research and educational purposes.

---

## 🙋 Contributing

Open to contributions:
- Add new algorithms (e.g., Tabu Search, Genetic)
- Add parallelization (OpenMP, CUDA)
- Improve benchmark visualization

Feel free to fork and submit a PR!

---
