#include <iostream>
#include <vector>
#include "knapsack.hpp"  // Assuming all knapsack classes are in this header

using namespace knapsack;

int main() {
    // Simple hardcoded data
    int capacity = 10;
    std::vector<Item> items = {
        {60, 10},  // value, weight
        {100, 20},
        {120, 30}
    };
    
    // 1. Zero-One Knapsack
    {
        ZeroOneKnapsack ks(capacity, items);
        std::cout << "Zero-One Knapsack:\n";
        std::cout << "DP: " << ks.solve("dp") << "\n";
        std::cout << "DP-1D: " << ks.solve("dp-1d") << "\n";
        std::cout << "Recursive-Memo: " << ks.solve("recursive-memo") << "\n";
        std::cout << "Brute-Force: " << ks.solve("brute-force") << "\n";
        std::cout << "Brute-Force-Bitmask: " << ks.solve("brute-force-bitmask") << "\n";
        std::cout << "Meet-in-Middle: " << ks.solve("meet-in-the-middle") << "\n";
        std::cout << "Branch-and-Bound: " << ks.solve("branch-and-bound") << "\n";
        std::cout << "FPTAS: " << ks.solve("fptas") << "\n";
        std::cout << "Greedy-Local: " << ks.solve("greedy-local") << "\n";
        std::cout << "DRL: " << ks.solve("drl") << "\n\n";
    }
    
    // 2. Fractional Knapsack
    {
        FractionalKnapsack ks(capacity);
        ks.items = items;
        std::cout << "Fractional Knapsack:\n";
        std::cout << "Greedy: " << ks.solve("greedy") << "\n";
        std::cout << "Sort-then-Fill: " << ks.solve("sort-then-fill") << "\n";
        std::cout << "Brute-Force: " << ks.solve("brute-force") << "\n";
        std::cout << "Brute-Force-Bitmask: " << ks.solve("brute-force-bitmask") << "\n\n";
    }
    
    // 3. Bounded Knapsack (with quantities)
    {
        std::vector<Item> boundedItems = {
            {60, 10, 2},  // value, weight, quantity
            {100, 20, 1},
            {120, 30, 3}
        };
        BoundedKnapsack ks(capacity);
        ks.items = boundedItems;
        std::cout << "Bounded Knapsack:\n";
        std::cout << "Brute-Force: " << ks.solve("brute-force") << "\n";
        std::cout << "Brute-Force-Bitmask: " << ks.solve("brute-force-bitmask") << "\n";
        std::cout << "Recursive-Memo: " << ks.solve("recursive-memo") << "\n";
        std::cout << "DP: " << ks.solve("dp") << "\n";
        std::cout << "Binary-Split-DP: " << ks.solve("binary-split-dp") << "\n";
        std::cout << "Branch-and-Bound: " << ks.solve("branch-and-bound") << "\n";
        std::cout << "Greedy-Local: " << ks.solve("greedy-local") << "\n\n";
    }
    
    // 4. Unbounded Knapsack
    {
        UnboundedKnapsack ks(capacity, items);
        std::cout << "Unbounded Knapsack:\n";
        std::cout << "Brute-Force: " << ks.solve("brute-force") << "\n";
        std::cout << "Recursive-Memo: " << ks.solve("recursive-memo") << "\n";
        std::cout << "DP: " << ks.solve("dp") << "\n";
        std::cout << "DP-1D: " << ks.solve("dp-1d") << "\n\n";
    }
    
    // 5. Multi-Dimensional Knapsack
    {
        std::vector<int> capacities = {10, 15};  // Two dimensions
        MultiDimensionalKnapsack ks(capacities, items);
        std::cout << "Multi-Dimensional Knapsack:\n";
        std::cout << "Brute-Force: " << ks.solve("brute-force") << "\n";
        std::cout << "Recursive-Memo: " << ks.solve("recursive-memo") << "\n";
        std::cout << "DP: " << ks.solve("dp") << "\n\n";
    }
    
    // 6. Multi-Objective Knapsack
    {
        std::vector<int> secondaryValues = {5, 10, 15};  // Secondary objectives
        MultiObjectiveKnapsack ks(capacity, items, secondaryValues);
        std::cout << "Multi-Objective Knapsack:\n";
        std::cout << "Brute-Force: " << ks.solve("brute-force") << "\n";
        std::cout << "DP: " << ks.solve("dp") << "\n";
        std::cout << "Lexicographic: " << ks.solve("lexicographic") << "\n";
        std::cout << "Weighted-Sum: " << ks.solve("weighted-sum") << "\n\n";
    }
    
    // 7. Multiple Knapsack
    {
        std::vector<int> bagCapacities = {10, 15};  // Two knapsacks
        MultipleKnapsack ks(bagCapacities, items);
        std::cout << "Multiple Knapsack:\n";
        std::cout << "Brute-Force: " << ks.solve("brute-force") << "\n";
        std::cout << "Greedy: " << ks.solve("greedy") << "\n";
        std::cout << "DP-Each-Bag: " << ks.solve("dp-each-bag") << "\n\n";
    }
    
    // 8. Quadratic Knapsack
    {
        QuadraticKnapsack ks(capacity, items);
        // Set interaction matrix
        std::vector<std::vector<int>> Q = {
            {0, 5, 3},
            {5, 0, 2},
            {3, 2, 0}
        };
        ks.setInteractionMatrix(Q);
        std::cout << "Quadratic Knapsack:\n";
        std::cout << "Brute-Force: " << ks.solve("brute-force") << "\n";
        std::cout << "Greedy: " << ks.solve("greedy") << "\n";
        std::cout << "DP-Approx: " << ks.solve("dp-approx") << "\n\n";
    }
    
    // 9. Stochastic Knapsack
    {
        StochasticKnapsack ks(capacity, items);
        // Set weight probability distributions
        ks.setWeightProb(0, {{8, 0.5}, {10, 0.3}, {12, 0.2}});
        ks.setWeightProb(1, {{18, 0.6}, {20, 0.4}});
        ks.setWeightProb(2, {{28, 0.7}, {30, 0.3}});
        std::cout << "Stochastic Knapsack:\n";
        std::cout << "Monte-Carlo: " << ks.solve("monte-carlo") << "\n";
        std::cout << "Greedy-Expected: " << ks.solve("greedy-expected") << "\n";
        std::cout << "Expected-DP: " << ks.solve("expected-dp") << "\n\n";
    }
    
    // 10. Multi-Choice Knapsack
    {
        std::vector<std::vector<Item>> groups = {
            {{60, 10}, {70, 12}},  // Group 1
            {{100, 20}, {110, 22}}, // Group 2
            {{120, 30}, {130, 32}}  // Group 3
        };
        MultiChoiceKnapsack ks(capacity, groups);
        std::cout << "Multi-Choice Knapsack:\n";
        std::cout << "Brute-Force: " << ks.solve("brute-force") << "\n";
        std::cout << "Greedy: " << ks.solve("greedy") << "\n";
        std::cout << "DP: " << ks.solve("dp") << "\n\n";
    }
    
    // 11. Metaheuristic Approaches
    {
        ZeroOneKnapsack ks(capacity, items);
        std::cout << "Metaheuristic Approaches:\n";
        std::cout << "Simulated-Annealing: " << MetaheuristicKnapsackSolver::solve(&ks, "simulated-annealing") << "\n";
        std::cout << "Ant-Colony: " << MetaheuristicKnapsackSolver::solve(&ks, "ant-colony") << "\n";
        std::cout << "PSO: " << MetaheuristicKnapsackSolver::solve(&ks, "pso") << "\n";
        std::cout << "ILP: " << MetaheuristicKnapsackSolver::solve(&ks, "ilp") << "\n";
        std::cout << "Constraint-Programming: " << MetaheuristicKnapsackSolver::solve(&ks, "constraint-programming") << "\n";
    }

    return 0;
}
