#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <random>
#include <queue>

struct Item {
    int value, weight, quantity;
    int count = 1;
    Item(int v, int w, int q = 1) : value(v), weight(w), quantity(q) {}
    double ratio() const { return (double)value / weight; }
};

class KnapsackBase {
public:
    std::vector<Item> items;
    int capacity;
    
    KnapsackBase(int cap, std::vector<Item> i) 
        : capacity(cap), items(std::move(i)) {}

    KnapsackBase(int cap) : capacity(cap) {}

    virtual std::string type() const = 0;
    virtual int solve(const std::string& algorithm) = 0;
};

class ZeroOneKnapsack : public KnapsackBase {
public:
    using KnapsackBase::KnapsackBase;
    std::string type() const override { return "ZeroOneKnapsack"; }

    int solve(const std::string& method) override {
        if (method == "dp") return solveDP();
        if (method == "dp-1d") return solveDP1D();
        if (method == "recursive-memo") return solveRecursiveMemo(0, capacity);
        if (method == "brute-force") return solveBruteForce(0, capacity);
        if (method == "brute-force-bitmask") return solveBruteBitmask();
        if (method == "meet-in-the-middle") return solveMeetInMiddle();
        if (method == "branch-and-bound") return solveBranchAndBound();
        if (method == "fptas") return solveFPTAS();
        if (method == "greedy-local") return solveGreedyLocalSearch();
        if (method == "drl") return solveDRL();        

        std::cerr << "Unsupported method for ZeroOneKnapsack";
        return -1;
    }

private:    
    int solveDP() {
        int n = items.size();
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));
        for (int i = 1; i <= n; ++i) {
            int wt = items[i - 1].weight;
            int val = items[i - 1].value;
            for (int w = 0; w <= capacity; ++w) {
                if (wt > w)
                    dp[i][w] = dp[i - 1][w];
                else
                    dp[i][w] = std::max(dp[i - 1][w], dp[i - 1][w - wt] + val);
            }
        }
        return dp[n][capacity];
    }
    
    int solveDP1D() {
        std::vector<int> dp(capacity + 1, 0);
        for (const auto& item : items) {
            for (int w = capacity; w >= item.weight; --w)
                dp[w] = std::max(dp[w], dp[w - item.weight] + item.value);
        }
        return dp[capacity];
    }

    int solveBruteForce(int idx, int remaining) {
        if (idx == items.size()) return 0;
        int without = solveBruteForce(idx + 1, remaining);
        int with = 0;
        if (items[idx].weight <= remaining)
            with = items[idx].value + solveBruteForce(idx + 1, remaining - items[idx].weight);
        return std::max(without, with);
    }
    
    int solveBruteBitmask() {
        int n = items.size(), best = 0;
        int limit = 1 << n;
        for (int mask = 0; mask < limit; ++mask) {
            int val = 0, wt = 0;
            for (int i = 0; i < n; ++i) {
                if (mask & (1 << i)) {
                    wt += items[i].weight;
                    val += items[i].value;
                }
            }
            if (wt <= capacity)
                best = std::max(best, val);
        }
        return best;
    }

    std::map<std::pair<int, int>, int> memo;
    int solveRecursiveMemo(int idx, int remaining) {
        if (idx == items.size()) return 0;
        auto key = std::make_pair(idx, remaining);
        if (memo.count(key)) return memo[key];

        int without = solveRecursiveMemo(idx + 1, remaining);
        int with = 0;
        if (items[idx].weight <= remaining)
            with = items[idx].value + solveRecursiveMemo(idx + 1, remaining - items[idx].weight);

        return memo[key] = std::max(without, with);
    }

    int solveMeetInMiddle() {
        int n = items.size();
        int half = n / 2;
        std::vector<std::pair<int, int>> A, B;

        auto generate = [](const std::vector<Item>& it, int start, int end, std::vector<std::pair<int, int>>& result) {
            int total = 1 << (end - start);
            for (int mask = 0; mask < total; ++mask) {
                int sumW = 0, sumV = 0;
                for (int i = 0; i < end - start; ++i) {
                    if (mask & (1 << i)) {
                        sumW += it[start + i].weight;
                        sumV += it[start + i].value;
                    }
                }
                result.push_back({sumW, sumV});
            }
        };

        generate(items, 0, half, A);
        generate(items, half, n, B);

        sort(B.begin(), B.end());
        std::vector<std::pair<int, int>> Bfiltered;
        int maxVal = -1;
        for (auto& p : B) {
            if (p.second > maxVal) {
                Bfiltered.push_back(p);
                maxVal = p.second;
            }
        }

        int best = 0;
        for (auto& a : A) {
            int remain = capacity - a.first;
            if (remain < 0) continue;

            int l = 0, r = Bfiltered.size() - 1, res = 0;
            while (l <= r) {
                int m = (l + r) / 2;
                if (Bfiltered[m].first <= remain) {
                    res = Bfiltered[m].second;
                    l = m + 1;
                } else {
                    r = m - 1;
                }
            }

            best = std::max(best, a.second + res);
        }
        return best;
    }

    int solveBranchAndBound() {
        struct Node {
            int level, profit, weight;
            double bound;
            Node(int l, int p, int w, double b) : level(l), profit(p), weight(w), bound(b) {}
        };

        struct CompareBound {
            bool operator()(const Node& a, const Node& b) {
                return a.bound < b.bound;  // max-heap
            }
        };

        auto bound = [&](const Node& u) -> double {
            if (u.weight >= capacity) return 0;
            double profit_bound = u.profit;
            int j = u.level + 1;
            int totweight = u.weight;

            while (j < items.size() && totweight + items[j].weight <= capacity) {
                totweight += items[j].weight;
                profit_bound += items[j].value;
                j++;
            }

            if (j < items.size())
                profit_bound += (capacity - totweight) * ((double)items[j].value / items[j].weight);

            return profit_bound;
        };

        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
            return (double)a.value / a.weight > (double)b.value / b.weight;
        });

        std::priority_queue<Node, std::vector<Node>, CompareBound> Q;
        Q.emplace(-1, 0, 0, bound(Node(-1, 0, 0, 0)));

        int maxProfit = 0;
        while (!Q.empty()) {
            Node u = Q.top(); Q.pop();
            if (u.bound <= maxProfit || u.level == (int)items.size() - 1)
                continue;

            Node v(u.level + 1, u.profit + items[u.level + 1].value,
                u.weight + items[u.level + 1].weight, 0);

            if (v.weight <= capacity && v.profit > maxProfit)
                maxProfit = v.profit;

            v.bound = bound(v);
            if (v.bound > maxProfit) Q.push(v);

            v.weight = u.weight;
            v.profit = u.profit;
            v.bound = bound(v);
            if (v.bound > maxProfit) Q.push(v);
        }

        return maxProfit;
    }

    int solveFPTAS(double epsilon = 0.1) {
        int n = items.size();
        if (n == 0) return 0;

        int maxVal = 0;
        for (const auto& item : items)
            maxVal = std::max(maxVal, item.value);

        double K = (epsilon * maxVal) / n;
        if (K < 1e-9) K = 1e-9;

        std::vector<int> scaled_values(n);
        int sum_scaled = 0;
        for (int i = 0; i < n; ++i) {
            scaled_values[i] = (int)(items[i].value / K);
            sum_scaled += scaled_values[i];
        }

        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(sum_scaled + 1, INT_MAX));
        dp[0][0] = 0;

        for (int i = 1; i <= n; ++i) {
            int sv = scaled_values[i - 1];
            int w = items[i - 1].weight;
            for (int j = 0; j <= sum_scaled; ++j) {
                dp[i][j] = dp[i - 1][j];
                if (j >= sv && dp[i - 1][j - sv] != INT_MAX)
                    dp[i][j] = std::min(dp[i][j], dp[i - 1][j - sv] + w);
            }
        }

        int best = 0;
        for (int j = 0; j <= sum_scaled; ++j)
            if (dp[n][j] <= capacity)
                best = j;

        return (int)(best * K);
    }

    int solveGreedyLocalSearch() {        
        std::vector<int> selected(items.size(), 0);
        std::vector<int> indices(items.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return (double)items[a].value / items[a].weight > (double)items[b].value / items[b].weight;
        });

        int totalWeight = 0, totalValue = 0;
        for (int i : indices) {
            if (totalWeight + items[i].weight <= capacity) {
                selected[i] = 1;
                totalWeight += items[i].weight;
                totalValue += items[i].value;
            }
        }
    
        bool improved = true;
        while (improved) {
            improved = false;
            for (size_t i = 0; i < items.size(); ++i) {
                if (selected[i]) {
                    for (size_t j = 0; j < items.size(); ++j) {
                        if (!selected[j]) {
                            int newWeight = totalWeight - items[i].weight + items[j].weight;
                            int newValue  = totalValue - items[i].value + items[j].value;
                            if (newWeight <= capacity && newValue > totalValue) {
                                selected[i] = 0;
                                selected[j] = 1;
                                totalWeight = newWeight;
                                totalValue  = newValue;
                                improved = true;
                            }
                        }
                    }
                }
            }
        }

        return totalValue;
    }

    int solveDRL() {        
        std::cout << "[INFO] solveDRL: This method would load a trained DRL agent and run inference.\n";
        std::cout << "[INFO] Simulating DRL decision-making...\n";

        // Random greedy approximation (just for structure demonstration)
        std::vector<int> indices(items.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

        int totalWeight = 0, totalValue = 0;
        for (int i : indices) {
            if (totalWeight + items[i].weight <= capacity) {
                totalWeight += items[i].weight;
                totalValue += items[i].value;
            }
        }

        return totalValue;
    }


};

class FractionalKnapsack : public KnapsackBase {
public:
    FractionalKnapsack(int cap) : KnapsackBase(cap) {}
    std::string type() const override { return "fractional"; }

    int solve(const std::string& algorithm) override {
        if (algorithm == "greedy") return solveGreedy();
        if (algorithm == "sort-then-fill") return solveSortThenFill();
        if (algorithm == "brute-force") return solveBruteForce(0, capacity);
        if (algorithm == "brute-force-bitmask") return solveBitmaskFraction();
        
        std::cout << "Unsupported algorithm for fractional knapsack.\n";
        return -1;
    }

private:    
    int solveGreedy() {
        sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
            return a.ratio() > b.ratio();
        });
        double total = 0;
        int W = capacity;
        for (auto& i : items) {
            if (W >= i.weight) {
                total += i.value;
                W -= i.weight;
            } else {
                total += i.ratio() * W;
                break;
            }
        }
        return (int)total;
    }
    
    int solveSortThenFill() {
        std::vector<Item> sortedItems = items;
        std::sort(sortedItems.begin(), sortedItems.end(), [](const Item& a, const Item& b) {
            return a.ratio() > b.ratio();
        });
        double total = 0;
        int W = capacity;
        for (auto& i : sortedItems) {
            if (W >= i.weight) {
                total += i.value;
                W -= i.weight;
            } else {
                total += i.ratio() * W;
                break;
            }
        }
        return (int)total;
    }
    
    double solveBruteForce(int idx, int remaining) {
        if (idx == items.size() || remaining == 0) return 0;

        double best = 0;
        int maxTake = std::min(items[idx].weight, remaining);

        // Try every fraction 0..maxTake (not efficient)
        for (int w = 0; w <= maxTake; ++w) {
            double takenValue = (double)items[idx].value * w / items[idx].weight;
            double rem = solveBruteForce(idx + 1, remaining - w);
            best = std::max(best, takenValue + rem);
        }

        return best;
    }
    
    int solveBitmaskFraction() {
        int n = items.size();
        double best = 0;
        int total = 1 << n;

        for (int mask = 0; mask < total; ++mask) {
            int weight = 0;
            double value = 0;
            std::vector<Item> remainingItems;

            for (int i = 0; i < n; ++i) {
                if (mask & (1 << i)) {
                    if (weight + items[i].weight <= capacity) {
                        weight += items[i].weight;
                        value += items[i].value;
                    }
                } else {
                    remainingItems.push_back(items[i]);
                }
            }

            int remCap = capacity - weight;
            if (remCap > 0 && !remainingItems.empty()) {
                sort(remainingItems.begin(), remainingItems.end(), [](const Item& a, const Item& b) {
                    return a.ratio() > b.ratio();
                });
                for (auto& i : remainingItems) {
                    if (remCap >= i.weight) {
                        value += i.value;
                        remCap -= i.weight;
                    } else {
                        value += i.ratio() * remCap;
                        break;
                    }
                }
            }

            best = std::max(best, value);
        }

        return (int)best;
    }
};

class BoundedKnapsack : public KnapsackBase {
public:
    BoundedKnapsack(int cap) : KnapsackBase(cap) {}
    std::string type() const override { return "bounded"; }

    int solve(const std::string& algorithm) override {
        if (algorithm == "brute-force") return solveBruteForce(0, capacity);
        if (algorithm == "brute-force-bitmask") return solveBruteForceBitmask();
        if (algorithm == "recursive-memo") return solveRecursiveMemo(0, capacity);
        if (algorithm == "dp") return solveDP();
        if (algorithm == "binary-split-dp") return solveBinarySplitDP();
        if (algorithm == "branch-and-bound") return solveBranchAndBound();
        if (algorithm == "greedy-local") return solveGreedyLocalSearch();

        std::cout << "Unsupported algorithm for Bounded Knapsack.\n";
        return -1;
    }

private:    
    int solveBruteForce(int idx, int remaining) {
        if (idx == items.size()) return 0;
        int maxVal = 0;
        for (int q = 0; q <= items[idx].quantity; ++q) {
            if (q * items[idx].weight <= remaining) {
                maxVal = std::max(maxVal, q * items[idx].value +
                    solveBruteForce(idx + 1, remaining - q * items[idx].weight));
            }
        }
        return maxVal;
    }
    
    int solveBruteForceBitmask() {
        std::vector<Item> expanded;
        for (auto& item : items) {
            for (int i = 0; i < item.quantity; ++i)
                expanded.push_back(Item(item.value, item.weight));
        }
        int n = expanded.size(), best = 0;
        int total = 1 << n;
        for (int mask = 0; mask < total; ++mask) {
            int val = 0, wt = 0;
            for (int i = 0; i < n; ++i) {
                if (mask & (1 << i)) {
                    val += expanded[i].value;
                    wt += expanded[i].weight;
                }
            }
            if (wt <= capacity)
                best = std::max(best, val);
        }
        return best;
    }
    
    std::map<std::pair<int, int>, int> memo;
    int solveRecursiveMemo(int idx, int remaining) {
        if (idx == items.size()) return 0;
        auto key = std::make_pair(idx, remaining);
        if (memo.count(key)) return memo[key];

        int best = 0;
        for (int q = 0; q <= items[idx].quantity; ++q) {
            if (q * items[idx].weight <= remaining) {
                best = std::max(best, q * items[idx].value +
                    solveRecursiveMemo(idx + 1, remaining - q * items[idx].weight));
            }
        }
        return memo[key] = best;
    }
    
    int solveDP() {
        int n = items.size();
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));

        for (int i = 1; i <= n; ++i) {
            int val = items[i - 1].value;
            int wt = items[i - 1].weight;
            int qty = items[i - 1].quantity;

            for (int w = 0; w <= capacity; ++w) {
                dp[i][w] = dp[i - 1][w];
                for (int q = 1; q <= qty; ++q) {
                    if (q * wt > w) break;
                    dp[i][w] = std::max(dp[i][w], dp[i - 1][w - q * wt] + q * val);
                }
            }
        }

        return dp[n][capacity];
    }
    
    int solveBinarySplitDP() {
        std::vector<Item> splitItems;
        for (auto& item : items) {
            int q = item.quantity;
            int k = 1;
            while (q > 0) {
                int take = std::min(k, q);
                splitItems.emplace_back(item.value * take, item.weight * take);
                q -= take;
                k *= 2;
            }
        }

        std::vector<int> dp(capacity + 1, 0);
        for (auto& item : splitItems) {
            for (int w = capacity; w >= item.weight; --w)
                dp[w] = std::max(dp[w], dp[w - item.weight] + item.value);
        }
        return dp[capacity];
    }

    int solveBranchAndBound() {
        struct Node {
            int level, profit, weight;
            double bound;
            std::vector<int> counts;
            Node(int l, int p, int w, double b, const std::vector<int>& c)
                : level(l), profit(p), weight(w), bound(b), counts(c) {}
        };

        struct Compare {
            bool operator()(const Node& a, const Node& b) {
                return a.bound < b.bound; // max-heap
            }
        };

        auto bound = [&](const Node& u) {
            if (u.weight >= capacity) return 0.0;
            double profit_bound = u.profit;
            int weight = u.weight;

            for (int i = u.level + 1; i < items.size(); ++i) {
                int take = std::min(items[i].quantity, (capacity - weight) / items[i].weight);
                weight += take * items[i].weight;
                profit_bound += take * items[i].value;

                if (weight < capacity)
                    profit_bound += (capacity - weight) * ((double)items[i].value / items[i].weight);
            }
            return profit_bound;
        };

        std::priority_queue<Node, std::vector<Node>, Compare> pq;
        std::vector<int> zero(items.size(), 0);
        pq.emplace(-1, 0, 0, bound(Node(-1, 0, 0, 0.0, zero)), zero);

        int maxProfit = 0;

        while (!pq.empty()) {
            Node u = pq.top(); pq.pop();
            if (u.level == (int)items.size() - 1) continue;

            int next = u.level + 1;
            for (int k = 0; k <= items[next].quantity; ++k) {
                int newWeight = u.weight + k * items[next].weight;
                int newProfit = u.profit + k * items[next].value;

                if (newWeight > capacity) break;

                std::vector<int> newCounts = u.counts;
                newCounts[next] = k;

                if (newProfit > maxProfit)
                    maxProfit = newProfit;

                double bnd = bound(Node(next, newProfit, newWeight, 0.0, newCounts));
                if (bnd > maxProfit) {
                    pq.emplace(next, newProfit, newWeight, bnd, newCounts);
                }
            }
        }

        return maxProfit;
    }

    int solveGreedyLocalSearch() {
        std::vector<int> indices(items.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return (double)items[a].value / items[a].weight > (double)items[b].value / items[b].weight;
        });

        int totalWeight = 0, totalValue = 0;
        std::vector<int> taken(items.size(), 0);

        for (int i : indices) {
            int maxTake = std::min(items[i].quantity, (capacity - totalWeight) / items[i].weight);
            totalWeight += maxTake * items[i].weight;
            totalValue += maxTake * items[i].value;
            taken[i] = maxTake;
        }

        return totalValue;
    }

};

class UnboundedKnapsack : public KnapsackBase {
public:
    using KnapsackBase::KnapsackBase;
    std::string type() const override { return "UnboundedKnapsack"; }

    int solve(const std::string& algorithm) override {
        if (algorithm == "brute-force") return solveBruteForce(capacity);
        if (algorithm == "recursive-memo") return solveRecursiveMemo(capacity);
        if (algorithm == "dp") return solveDP();
        if (algorithm == "dp-1d") return solveDP1D();

        std::cout << "Unsupported algorithm for Unbounded Knapsack.\n";
        return -1;
    }

private:    
    int solveBruteForce(int remaining) {
        if (remaining == 0) return 0;
        int best = 0;
        for (const auto& item : items) {
            if (item.weight <= remaining) {
                best = std::max(best, item.value + solveBruteForce(remaining - item.weight));
            }
        }
        return best;
    }
    
    std::unordered_map<int, int> memo;
    int solveRecursiveMemo(int remaining) {
        if (remaining == 0) return 0;
        if (memo.count(remaining)) return memo[remaining];

        int best = 0;
        for (const auto& item : items) {
            if (item.weight <= remaining) {
                best = std::max(best, item.value + solveRecursiveMemo(remaining - item.weight));
            }
        }
        return memo[remaining] = best;
    }
    
    int solveDP() {
        int n = items.size();
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));
        for (int i = 1; i <= n; ++i) {
            int wt = items[i - 1].weight;
            int val = items[i - 1].value;
            for (int w = 0; w <= capacity; ++w) {
                dp[i][w] = dp[i - 1][w];
                if (w >= wt)
                    dp[i][w] = std::max(dp[i][w], dp[i][w - wt] + val);
            }
        }
        return dp[n][capacity];
    }
    
    int solveDP1D() {
        std::vector<int> dp(capacity + 1, 0);
        for (int w = 0; w <= capacity; ++w) {
            for (const auto& item : items) {
                if (item.weight <= w) {
                    dp[w] = std::max(dp[w], dp[w - item.weight] + item.value);
                }
            }
        }
        return dp[capacity];
    }
};

class MultiDimensionalKnapsack : public KnapsackBase {
public:
    std::vector<int> capacities;

    MultiDimensionalKnapsack(std::vector<int> caps, std::vector<Item> i)
        : KnapsackBase(caps[0]), capacities(std::move(caps)) {
        items = std::move(i);
    }

    std::string type() const override { return "MultiDimensionalKnapsack"; }

    int solve(const std::string& algorithm) override {
        if (algorithm == "brute-force") return solveBruteForce(0, capacities);
        if (algorithm == "recursive-memo") return solveRecursiveMemo(0, capacities);
        if (algorithm == "dp") return solveDP();

        std::cout << "Unsupported algorithm for MultiDimensional Knapsack.\n";
        return -1;
    }

private:    
    int solveBruteForce(int idx, std::vector<int> remaining) {
        if (idx == items.size()) return 0;

        int without = solveBruteForce(idx + 1, remaining);
        bool canTake = true;
        for (size_t d = 0; d < remaining.size(); ++d) {
            if (items[idx].weight > remaining[d]) {
                canTake = false;
                break;
            }
        }

        int with = 0;
        if (canTake) {
            std::vector<int> updated = remaining;
            for (size_t d = 0; d < updated.size(); ++d)
                updated[d] -= items[idx].weight;
            with = items[idx].value + solveBruteForce(idx + 1, updated);
        }

        return std::max(without, with);
    }
    
    std::map<std::tuple<int, std::vector<int>>, int> memo;
    int solveRecursiveMemo(int idx, std::vector<int> remaining) {
        if (idx == items.size()) return 0;
        auto key = std::make_tuple(idx, remaining);
        if (memo.count(key)) return memo[key];

        int without = solveRecursiveMemo(idx + 1, remaining);
        bool canTake = true;
        for (size_t d = 0; d < remaining.size(); ++d) {
            if (items[idx].weight > remaining[d]) {
                canTake = false;
                break;
            }
        }

        int with = 0;
        if (canTake) {
            std::vector<int> updated = remaining;
            for (size_t d = 0; d < updated.size(); ++d)
                updated[d] -= items[idx].weight;
            with = items[idx].value + solveRecursiveMemo(idx + 1, updated);
        }

        return memo[key] = std::max(without, with);
    }
    
    int solveDP() {
        int n = items.size();
        int D = capacities.size();

        if (D == 2) {
            int C1 = capacities[0], C2 = capacities[1];
            std::vector<std::vector<int>> dp(C1 + 1, std::vector<int>(C2 + 1, 0));
            for (int k = 0; k < n; ++k) {
                for (int i = C1; i >= items[k].weight; --i) {
                    for (int j = C2; j >= items[k].weight; --j) {
                        dp[i][j] = std::max(dp[i][j], dp[i - items[k].weight][j - items[k].weight] + items[k].value);
                    }
                }
            }
            return dp[C1][C2];
        }

        if (D == 1) {
            int C = capacities[0];
            std::vector<int> dp(C + 1, 0);
            for (int k = 0; k < n; ++k) {
                for (int i = C; i >= items[k].weight; --i) {
                    dp[i] = std::max(dp[i], dp[i - items[k].weight] + items[k].value);
                }
            }
            return dp[C];
        }

        std::cout << "Too many dimensions for DP (>2).\n";
        return -1;
    }
};

class MultiObjectiveKnapsack : public KnapsackBase {
public:
    std::vector<int> secondaryValues;

    MultiObjectiveKnapsack(int c, std::vector<Item> i, std::vector<int> secVals)
        : KnapsackBase(c), secondaryValues(std::move(secVals)) {
        items = std::move(i);
    }

    std::string type() const override { return "MultiObjectiveKnapsack"; }

    int solve(const std::string& algorithm) override {
        if (algorithm == "brute-force") return solveBruteForce();
        if (algorithm == "dp") return solveParetoDP();
        if (algorithm == "lexicographic") return solveLexicographic();
        if (algorithm == "weighted-sum") return solveWeightedSum(1.0);  // Default λ = 1.0

        std::cout << "Unsupported algorithm for MultiObjectiveKnapsack.\n";
        return -1;
    }

private:
    struct Solution {
        int totalWeight;
        int primary;
        int secondary;
    };

    bool dominates(const Solution& a, const Solution& b) {
        return a.primary >= b.primary && a.secondary >= b.secondary &&
               (a.primary > b.primary || a.secondary > b.secondary);
    }
    
    int solveBruteForce() {
        int n = items.size();
        int bestPrimary = 0;
        std::vector<Solution> pareto;

        for (int mask = 0; mask < (1 << n); ++mask) {
            int w = 0, v1 = 0, v2 = 0;
            for (int i = 0; i < n; ++i) {
                if (mask & (1 << i)) {
                    w += items[i].weight;
                    v1 += items[i].value;
                    v2 += secondaryValues[i];
                }
            }
            if (w <= capacity) {
                Solution sol = {w, v1, v2};
                bool dominated = false;
                for (const auto& p : pareto) {
                    if (dominates(p, sol)) {
                        dominated = true;
                        break;
                    }
                }
                if (!dominated) {
                    pareto.push_back(sol);
                    bestPrimary = std::max(bestPrimary, v1);
                }
            }
        }
        return bestPrimary;
    }
    
    int solveLexicographic() {
        int n = items.size();
        int bestPrimary = 0, bestSecondary = 0;
        for (int mask = 0; mask < (1 << n); ++mask) {
            int w = 0, v1 = 0, v2 = 0;
            for (int i = 0; i < n; ++i) {
                if (mask & (1 << i)) {
                    w += items[i].weight;
                    v1 += items[i].value;
                    v2 += secondaryValues[i];
                }
            }
            if (w <= capacity) {
                if (v1 > bestPrimary || (v1 == bestPrimary && v2 > bestSecondary)) {
                    bestPrimary = v1;
                    bestSecondary = v2;
                }
            }
        }
        return bestPrimary;
    }
    
    int solveWeightedSum(double lambda) {
        int n = items.size();
        double bestScore = 0;
        int bestPrimary = 0;
        for (int mask = 0; mask < (1 << n); ++mask) {
            int w = 0, v1 = 0, v2 = 0;
            for (int i = 0; i < n; ++i) {
                if (mask & (1 << i)) {
                    w += items[i].weight;
                    v1 += items[i].value;
                    v2 += secondaryValues[i];
                }
            }
            if (w <= capacity) {
                double score = v1 + lambda * v2;
                if (score > bestScore) {
                    bestScore = score;
                    bestPrimary = v1;
                }
            }
        }
        return bestPrimary;
    }
    
    int solveParetoDP() {
        int n = items.size();
        std::vector<std::vector<Solution>> dp(capacity + 1);
        dp[0].push_back({0, 0, 0});

        for (int i = 0; i < n; ++i) {
            int wt = items[i].weight;
            int v1 = items[i].value;
            int v2 = secondaryValues[i];
            for (int w = capacity; w >= wt; --w) {
                for (const auto& s : dp[w - wt]) {
                    Solution newSol = {w, s.primary + v1, s.secondary + v2};
                    bool dominated = false;
                    for (const auto& existing : dp[w]) {
                        if (dominates(existing, newSol)) {
                            dominated = true;
                            break;
                        }
                    }
                    if (!dominated) {
                        dp[w].push_back(newSol);
                    }
                }
            }
        }

        int best = 0;
        for (const auto& solutions : dp) {
            for (const auto& s : solutions) {
                best = std::max(best, s.primary);
            }
        }
        return best;
    }
};

class MultipleKnapsack : public KnapsackBase {
public:
    std::vector<int> bagCapacities;

    MultipleKnapsack(std::vector<int> caps, std::vector<Item> i)
        : KnapsackBase(0, i), bagCapacities(std::move(caps)) {}

    std::string type() const override { return "MultipleKnapsack"; }

    int solve(const std::string& algorithm) override {
        if (algorithm == "brute-force") return solveBruteForce(0, bagCapacities, 0);
        if (algorithm == "greedy") return solveGreedy();
        if (algorithm == "dp-each-bag") return solveDPEachBag();

        std::cout << "Unsupported algorithm for MultipleKnapsack.\n";
        return -1;
    }

private:    
    int solveBruteForce(int idx, std::vector<int> bags, int currentValue) {
        if (idx == items.size()) return currentValue;

        int best = currentValue;
        for (size_t b = 0; b < bags.size(); ++b) {
            if (bags[b] >= items[idx].weight) {
                bags[b] -= items[idx].weight;
                best = std::max(best, solveBruteForce(idx + 1, bags, currentValue + items[idx].value));
                bags[b] += items[idx].weight;
            }
        }    
        best = std::max(best, solveBruteForce(idx + 1, bags, currentValue));
        return best;
    }
    
    int solveGreedy() {
        std::vector<Item> sorted = items;
        std::sort(sorted.begin(), sorted.end(), [](const Item& a, const Item& b) {
            return a.ratio() > b.ratio();
        });

        std::vector<int> remaining = bagCapacities;
        int totalValue = 0;

        for (const auto& item : sorted) {
            int bestBag = -1, minSpace = INT_MAX;
            for (size_t i = 0; i < remaining.size(); ++i) {
                if (remaining[i] >= item.weight && remaining[i] < minSpace) {
                    bestBag = i;
                    minSpace = remaining[i];
                }
            }
            if (bestBag != -1) {
                remaining[bestBag] -= item.weight;
                totalValue += item.value;
            }
        }
        return totalValue;
    }
    
    int solveDPEachBag() {
        int totalValue = 0;
        std::vector<bool> used(items.size(), false);

        for (int cap : bagCapacities) {
            int n = items.size();
            std::vector<std::vector<int>> dp(n + 1, std::vector<int>(cap + 1, 0));

            for (int i = 1; i <= n; ++i) {
                for (int w = 0; w <= cap; ++w) {
                    dp[i][w] = dp[i - 1][w];
                    if (!used[i - 1] && w >= items[i - 1].weight)
                        dp[i][w] = std::max(dp[i][w], dp[i - 1][w - items[i - 1].weight] + items[i - 1].value);
                }
            }
            
            int w = cap;
            for (int i = n; i >= 1; --i) {
                if (dp[i][w] != dp[i - 1][w]) {
                    used[i - 1] = true;
                    w -= items[i - 1].weight;
                    totalValue += items[i - 1].value;
                }
            }
        }

        return totalValue;
    }
};

class QuadraticKnapsack : public KnapsackBase {
public:
    std::vector<std::vector<int>> Q;

    QuadraticKnapsack(int c, std::vector<Item> i)
        : KnapsackBase(c) {
        items = std::move(i);
        int n = items.size();
        Q = std::vector<std::vector<int>>(n, std::vector<int>(n, 0)); 
    }

    std::string type() const override { return "QuadraticKnapsack"; }

    int solve(const std::string& algorithm) override {
        if (algorithm == "brute-force") return solveBruteForce();
        if (algorithm == "greedy") return solveGreedyIgnoreQ();
        if (algorithm == "dp-approx") return solveDPIgnoreQ();

        std::cout << "Unsupported algorithm for QuadraticKnapsack.\n";
        return -1;
    }

    void setInteractionMatrix(const std::vector<std::vector<int>>& q) {
        Q = q;
    }

private:    
    int solveBruteForce() {
        int n = items.size();
        int best = 0;

        for (int mask = 0; mask < (1 << n); ++mask) {
            int weight = 0;
            int value = 0;

            std::vector<int> chosen;
            for (int i = 0; i < n; ++i) {
                if (mask & (1 << i)) {
                    weight += items[i].weight;
                    value += items[i].value;
                    chosen.push_back(i);
                }
            }

            if (weight > capacity) continue;
            
            for (size_t i = 0; i < chosen.size(); ++i)
                for (size_t j = i + 1; j < chosen.size(); ++j)
                    value += Q[chosen[i]][chosen[j]];

            best = std::max(best, value);
        }

        return best;
    }
    
    int solveGreedyIgnoreQ() {
        std::vector<Item> sorted = items;
        std::sort(sorted.begin(), sorted.end(), [](const Item& a, const Item& b) {
            return a.ratio() > b.ratio();
        });

        int W = capacity, value = 0;
        for (const auto& i : sorted) {
            if (W >= i.weight) {
                W -= i.weight;
                value += i.value;
            }
        }
        return value;
    }
    
    int solveDPIgnoreQ() {
        int n = items.size();
        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));

        for (int i = 1; i <= n; ++i) {
            int wt = items[i - 1].weight;
            int val = items[i - 1].value;
            for (int w = 0; w <= capacity; ++w) {
                if (wt > w)
                    dp[i][w] = dp[i - 1][w];
                else
                    dp[i][w] = std::max(dp[i - 1][w], dp[i - 1][w - wt] + val);
            }
        }

        return dp[n][capacity];
    }
};

class StochasticKnapsack : public KnapsackBase {
public:
    std::vector<std::vector<std::pair<int, double>>> weightProb; 

    StochasticKnapsack(int cap, std::vector<Item> i)
        : KnapsackBase(cap) {
        items = std::move(i);
        weightProb.resize(items.size()); 
    }

    std::string type() const override { return "StochasticKnapsack"; }

    int solve(const std::string& algorithm) override {
        if (algorithm == "monte-carlo") return solveMonteCarlo(10000);
        if (algorithm == "greedy-expected") return solveGreedyExpected();
        if (algorithm == "expected-dp") return solveExpectedDP();

        std::cout << "Unsupported algorithm for StochasticKnapsack.\n";
        return -1;
    }

    void setWeightProb(int index, const std::vector<std::pair<int, double>>& dist) {
        weightProb[index] = dist;
    }

private:    
    int solveMonteCarlo(int trials) {
        int bestValue = 0;
        int n = items.size();

        std::default_random_engine gen(std::random_device{}());

        for (int t = 0; t < trials; ++t) {
            std::vector<int> sampledWeights(n);
            for (int i = 0; i < n; ++i) {
                double p = std::generate_canonical<double, 10>(gen);
                double sum = 0.0;
                for (auto& wp : weightProb[i]) {
                    sum += wp.second;
                    if (p <= sum) {
                        sampledWeights[i] = wp.first;
                        break;
                    }
                }
            }
            
            std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));
            for (int i = 1; i <= n; ++i) {
                for (int w = 0; w <= capacity; ++w) {
                    if (sampledWeights[i - 1] > w)
                        dp[i][w] = dp[i - 1][w];
                    else
                        dp[i][w] = std::max(dp[i - 1][w], dp[i - 1][w - sampledWeights[i - 1]] + items[i - 1].value);
                }
            }
            bestValue = std::max(bestValue, dp[n][capacity]);
        }
        return bestValue;
    }
    
    int solveGreedyExpected() {
        int n = items.size();
        std::vector<std::tuple<double, int>> sorted;

        for (int i = 0; i < n; ++i) {
            double expectedW = 0.0;
            for (auto& [w, p] : weightProb[i]) expectedW += w * p;
            if (expectedW == 0) continue;
            sorted.emplace_back((double)items[i].value / expectedW, i);
        }

        std::sort(sorted.begin(), sorted.end(), std::greater<>());

        int W = capacity, totalValue = 0;
        for (auto& [_, idx] : sorted) {
            double expectedW = 0;
            for (auto& [w, p] : weightProb[idx]) expectedW += w * p;
            int ew = (int)std::round(expectedW);
            if (W >= ew) {
                W -= ew;
                totalValue += items[idx].value;
            }
        }

        return totalValue;
    }
    
    int solveExpectedDP() {
        int n = items.size();
        std::vector<int> expectedWeights(n);

        for (int i = 0; i < n; ++i) {
            double ew = 0.0;
            for (auto& [w, p] : weightProb[i]) ew += w * p;
            expectedWeights[i] = (int)std::round(ew);
        }

        std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));
        for (int i = 1; i <= n; ++i) {
            int wt = expectedWeights[i - 1];
            int val = items[i - 1].value;
            for (int w = 0; w <= capacity; ++w) {
                if (wt > w)
                    dp[i][w] = dp[i - 1][w];
                else
                    dp[i][w] = std::max(dp[i - 1][w], dp[i - 1][w - wt] + val);
            }
        }

        return dp[n][capacity];
    }
};

class MultiChoiceKnapsack : public KnapsackBase {
public:
    std::vector<std::vector<Item>> groups;

    MultiChoiceKnapsack(int c, std::vector<std::vector<Item>> g)
        : KnapsackBase(c, {}), groups(std::move(g)) {}

    std::string type() const override { return "MultiChoiceKnapsack"; }

    int solve(const std::string& algorithm) override {
        if (algorithm == "brute-force") return solveBruteForce(0, 0, 0);
        if (algorithm == "greedy") return solveGreedy();
        if (algorithm == "dp") return solveDP();

        std::cout << "Unsupported algorithm for MultiChoiceKnapsack.\n";
        return -1;
    }

private:    
    int solveBruteForce(int groupIndex, int totalWeight, int totalValue) {
        if (groupIndex == groups.size()) {
            return totalWeight <= capacity ? totalValue : 0;
        }

        int best = 0;
        for (const Item& item : groups[groupIndex]) {
            if (totalWeight + item.weight <= capacity) {
                best = std::max(best, solveBruteForce(
                    groupIndex + 1,
                    totalWeight + item.weight,
                    totalValue + item.value
                ));
            } else {                
                best = std::max(best, solveBruteForce(
                    groupIndex + 1,
                    totalWeight,
                    totalValue
                ));
            }
        }
        return best;
    }
    
    int solveGreedy() {
        int totalWeight = 0;
        int totalValue = 0;

        for (auto& group : groups) {
            auto bestIt = std::max_element(group.begin(), group.end(), [](const Item& a, const Item& b) {
                return a.ratio() < b.ratio();
            });

            if (totalWeight + bestIt->weight <= capacity) {
                totalWeight += bestIt->weight;
                totalValue += bestIt->value;
            }
        }

        return totalValue;
    }
    
    int solveDP() {
        int G = groups.size();
        std::vector<std::vector<int>> dp(G + 1, std::vector<int>(capacity + 1, 0));

        for (int g = 1; g <= G; ++g) {
            for (int w = 0; w <= capacity; ++w) {
                dp[g][w] = dp[g - 1][w]; 
                for (const auto& item : groups[g - 1]) {
                    if (w >= item.weight) {
                        dp[g][w] = std::max(dp[g][w], dp[g - 1][w - item.weight] + item.value);
                    }
                }
            }
        }

        return dp[G][capacity];
    }
};

class MetaheuristicKnapsackSolver {
public:
    static int solve(KnapsackBase* knapsack, const std::string& algorithm) {
        if (algorithm == "simulated-annealing") return solveSimulatedAnnealing(knapsack);
        if (algorithm == "ant-colony") return solveAntColony(knapsack);
        if (algorithm == "pso") return solveParticleSwarm(knapsack);
        if (algorithm == "ilp") return solveILP(knapsack);
        if (algorithm == "constraint-programming") return solveCP(knapsack);

        std::cout << "Unsupported global algorithm.\n";
        return -1;
    }

private:
    static int evaluate(KnapsackBase* knapsack, const std::vector<int>& selected) {
        int totalWeight = 0, totalValue = 0;
        for (size_t i = 0; i < knapsack->items.size(); ++i) {
            totalWeight += knapsack->items[i].weight * selected[i];
            totalValue  += knapsack->items[i].value  * selected[i];
        }
        return (totalWeight <= knapsack->capacity) ? totalValue : 0;
    }

    static int solveSimulatedAnnealing(KnapsackBase* knapsack) {
        int n = knapsack->items.size();
        std::vector<int> current(n, 0), best(n, 0);
        int bestVal = 0;

        double temp = 1000, cooling = 0.95;
        std::mt19937 rng(std::random_device{}());

        for (int iter = 0; iter < 1000; ++iter) {
            std::vector<int> neighbor = current;
            int i = rng() % n;
            neighbor[i] = 1 - neighbor[i];

            int val = evaluate(knapsack, neighbor);
            if (val > bestVal || std::exp((val - bestVal) / temp) > (double)rng() / rng.max()) {
                current = neighbor;
                bestVal = val;
                best = neighbor;
            }
            temp *= cooling;
        }
        return bestVal;
    }

    static int solveAntColony(KnapsackBase* knapsack) {
        int n = knapsack->items.size();
        std::vector<double> pheromone(n, 1.0);
        int bestVal = 0;
        std::vector<int> best;

        std::mt19937 rng(std::random_device{}());

        for (int iter = 0; iter < 50; ++iter) {
            std::vector<int> candidate(n, 0);
            for (int i = 0; i < n; ++i)
                candidate[i] = (rng() / (double)rng.max()) < pheromone[i] ? 1 : 0;

            int val = evaluate(knapsack, candidate);
            if (val > bestVal) {
                bestVal = val;
                best = candidate;
            }
            
            for (int i = 0; i < n; ++i)
                pheromone[i] = 0.9 * pheromone[i] + 0.1 * candidate[i];
        }
        return bestVal;
    }

    static int solveParticleSwarm(KnapsackBase* knapsack) {
        int n = knapsack->items.size();
        std::vector<int> bestGlobal(n, 0);
        int bestValue = 0;
        std::mt19937 rng(std::random_device{}());

        for (int iter = 0; iter < 100; ++iter) {
            std::vector<int> particle(n);
            for (int i = 0; i < n; ++i)
                particle[i] = rng() % 2;

            int val = evaluate(knapsack, particle);
            if (val > bestValue) {
                bestValue = val;
                bestGlobal = particle;
            }
        }
        return bestValue;
    }

    static int solveILP(KnapsackBase* knapsack) {
        std::cout << "[ILP Placeholder] Ideally solved using CPLEX, Gurobi, or OR-Tools.\n";
        return solveSimulatedAnnealing(knapsack);
    }

    static int solveCP(KnapsackBase* knapsack) {
        std::cout << "[Constraint Programming Placeholder] Use Google OR-Tools or similar.\n";
        return solveAntColony(knapsack);
    }
};
