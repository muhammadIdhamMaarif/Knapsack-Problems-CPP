    #include <iostream>
    #include <vector>
    #include <string>
    #include <map>
    #include <algorithm>
    #include <unordered_map>
    #include <set>
    #include <random>
    #include <queue>
    #include <limits>
    #include <climits>
    #include <random>
    #include <chrono>
    #include <iomanip>
    #include <thread>
    #include <atomic>
    #include <sstream>
    #include <streambuf>
    #include <fstream>
    #include <filesystem>
    #include <csignal>

    // Variabel untuk iterasi eksperimen
    #define MAX_NUMBER_ITEMS_TESTED     7
    #define MAX_ITEMS_VALUE_TESTED      700     
    #define MAX_ITEMS_WEIGHT_TESTED     700    
    #define MAX_QUANTITY_TESTED         7           
    #define MAX_CAPACITY_TESTED         700        

    // Increment untuk iterasi eksperimen
    #define STEP_NUMBER_ITEMS           1
    #define STEP_ITEMS_VALUE            100
    #define STEP_ITEMS_WEIGHT           100
    #define STEP_QUANTITY               1
    #define STEP_CAPACITY               100

    // Nama untuk penyimpanan file
    // "Algorithm,TotalItems,ItemsValue,ItemsWeight,Quantity,Capacity,Time_ns\n"
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

    //other
    #define DEFAULT_CAPACITY_TESTED     50
    #define DISABLE_BRUTE_FORCE         true

    namespace knapsack {

    // Struktur data untuk merepresentasikan sebuah item pada knapsack
    struct Item {
        int value, weight, quantity;
        int count = 1;

        // Default constructor
        Item() : value(0), weight(0), quantity(1) {}
        
        // Konstruktor: inisialisasi item dengan value, weight, dan quantity
        Item(int v, int w, int q = 1) : value(v), weight(w), quantity(q) {}

        // Fungsi pembantu: menghitung rasio value/weight dari item (untuk greedy heuristic)
        double ratio() const { return (double)value / weight; }
    };

    // Kelas dasar abstrak untuk semua jenis knapsack
    class KnapsackBase {
    public:
        std::vector<Item> items;  // Daftar item yang tersedia
        int capacity;             // Kapasitas maksimum knapsack

        // Konstruktor dengan parameter: kapasitas dan daftar item
        KnapsackBase(int cap, std::vector<Item> i) 
            : capacity(cap), items(std::move(i)) {}

        // Konstruktor hanya dengan kapasitas, item dapat ditambahkan belakangan
        KnapsackBase(int cap) : capacity(cap) {}

        // Fungsi virtual murni: untuk mengidentifikasi tipe knapsack (harus di-override)
        virtual std::string type() const = 0;

        // Fungsi virtual murni: menyelesaikan masalah knapsack menggunakan algoritma tertentu
        virtual int solve(const std::string& algorithm) = 0;
    };

    // Kelas untuk knapsack 0/1 (Zero-One Knapsack Problem)
    class ZeroOneKnapsack : public KnapsackBase {
    public:
        using KnapsackBase::KnapsackBase; // Mewarisi konstruktor dari KnapsackBase

        // Override fungsi type() untuk mengembalikan nama tipe
        std::string type() const override { return "ZeroOneKnapsack"; }

        // Override fungsi solve() untuk memilih metode penyelesaian berdasarkan parameter
        int solve(const std::string& method = "dp") override {
            if (method == "dp") return solveDP();                                   // Dynamic Programming (2D)
            if (method == "dp-1d") return solveDP1D();                              // DP versi 1 dimensi
            if (method == "recursive-memo") return solveRecursiveMemo(0, capacity); // Rekursif + Memoisasi            
            if (method == "brute-force" && !DISABLE_BRUTE_FORCE) return solveBruteForce(0, capacity);       // Brute-force rekursif
            if (method == "brute-force-bitmask" && !DISABLE_BRUTE_FORCE) return solveBruteBitmask();        // Bitmasking
            if (method == "meet-in-the-middle") return solveMeetInMiddle();         // Meet in the Middle
            if (method == "branch-and-bound") return solveBranchAndBound();         // Branch and Bound
            if (method == "fptas") return solveFPTAS();                             // Approximation Scheme
            if (method == "greedy-local") return solveGreedyLocalSearch();          // Heuristic Greedy + Local Search
            // if (method == "simulated-annealing") MetaheuristicKnapsackSolver::solve("simulated-annealing");
            // if (method == "drl") return solveDRL();                                 // Deep Reinforcement Learning

            // Menampilkan pesan error jika metode tidak dikenali
            std::cerr << "Unsupported method for ZeroOneKnapsack";
            return -1;

            // "simulated-annealing", "ant-colony", "pso", "ilp", "constraint-programming"
        }

    private:
        // Dynamic Programming klasik (2D) untuk Zero-One Knapsack
        int solveDP() {
            int n = items.size(); // Jumlah item
            // Membuat tabel dp[n+1][capacity+1] dan inisialisasi ke 0
            std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));

            // Iterasi untuk setiap item
            for (int i = 1; i <= n; ++i) {
                int wt = items[i - 1].weight; // Berat item ke-i
                int val = items[i - 1].value; // Nilai item ke-i
                for (int w = 0; w <= capacity; ++w) {
                    if (wt > w)
                        // Jika item terlalu berat, tidak bisa dimasukkan
                        dp[i][w] = dp[i - 1][w];
                    else
                        // Pilih maksimum antara tidak mengambil atau mengambil item
                        dp[i][w] = std::max(dp[i - 1][w], dp[i - 1][w - wt] + val);
                }
            }

            // Nilai maksimum yang bisa diambil dengan kapasitas penuh
            return dp[n][capacity];
        }
        
        // Dynamic Programming dengan optimasi space (menggunakan array 1 dimensi)
        int solveDP1D() {
            std::vector<int> dp(capacity + 1, 0); // dp[w] menyimpan nilai maksimum dengan kapasitas w
            for (const auto& item : items) {
                // Iterasi mundur agar nilai dp[w - item.weight] masih relevan untuk iterasi saat ini
                for (int w = capacity; w >= item.weight; --w)
                    dp[w] = std::max(dp[w], dp[w - item.weight] + item.value);
            }
            return dp[capacity]; // Nilai maksimum pada kapasitas penuh
        }

        // Brute-force rekursif: mencoba semua kemungkinan dengan dan tanpa mengambil item
        int solveBruteForce(int idx, int remaining) {
            if (idx == items.size()) return 0; // Basis: tidak ada item tersisa

            // Tidak ambil item ke-idx
            int without = solveBruteForce(idx + 1, remaining);

            int with = 0;
            // Ambil item jika masih cukup kapasitas
            if (items[idx].weight <= remaining)
                with = items[idx].value + solveBruteForce(idx + 1, remaining - items[idx].weight);

            return std::max(without, with); // Ambil solusi terbaik
        }

        // Brute-force menggunakan bitmasking untuk mengecek semua kombinasi subset item
        int solveBruteBitmask() {
            int n = items.size(), best = 0;
            int limit = 1 << n; // Total kemungkinan kombinasi = 2^n

            for (int mask = 0; mask < limit; ++mask) {
                int val = 0, wt = 0;
                for (int i = 0; i < n; ++i) {
                    if (mask & (1 << i)) { // Item i diambil dalam subset ini
                        wt += items[i].weight;
                        val += items[i].value;
                    }
                }
                if (wt <= capacity)
                    best = std::max(best, val); // Simpan nilai terbaik yang valid
            }
            return best;
        }

        // Memoisasi untuk menghindari perhitungan ulang submasalah
        std::map<std::pair<int, int>, int> memo;
        int solveRecursiveMemo(int idx, int remaining) {
            if (idx == items.size()) return 0; // Basis

            auto key = std::make_pair(idx, remaining);
            if (memo.count(key)) return memo[key]; // Cek cache

            int without = solveRecursiveMemo(idx + 1, remaining); // Tidak ambil item
            int with = 0;
            if (items[idx].weight <= remaining)
                with = items[idx].value + solveRecursiveMemo(idx + 1, remaining - items[idx].weight); // Ambil item

            return memo[key] = std::max(without, with); // Simpan hasil dalam memo
        }

        // Meet-in-the-middle: membagi set item menjadi dua bagian dan kombinasikan hasilnya
        int solveMeetInMiddle() {
            int n = items.size();
            int half = n / 2;
            std::vector<std::pair<int, int>> A, B;

            // Fungsi bantu: generate semua subset antara start dan end, simpan (weight, value)
            auto generate = [](const std::vector<Item>& it, int start, int end, std::vector<std::pair<int, int>>& result) {
                int total = 1 << (end - start); // Total kombinasi subset
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

            // Urutkan subset bagian kedua berdasarkan berat
            sort(B.begin(), B.end());

            // Filter B agar tidak ada pasangan (w1, v1), (w2, v2) di mana w2 > w1 dan v2 <= v1
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

                // Binary search nilai terbaik dari Bfiltered untuk kapasitas sisa
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

        // Branch and Bound: mengeksplorasi subset dengan prioritas pada node yang paling menjanjikan
        int solveBranchAndBound() {
            struct Node {
                int level, profit, weight;
                double bound; // Estimasi maksimum profit yang bisa dicapai dari node ini
                Node(int l, int p, int w, double b) : level(l), profit(p), weight(w), bound(b) {}
            };

            // Membuat max-heap berdasarkan nilai bound tertinggi
            struct CompareBound {
                bool operator()(const Node& a, const Node& b) {
                    return a.bound < b.bound;
                }
            };

            // Fungsi estimasi upper bound dari profit maksimal untuk node u
            auto bound = [&](const Node& u) -> double {
                if (u.weight >= capacity) return 0; // Tidak valid jika melebihi kapasitas

                double profit_bound = u.profit;
                int j = u.level + 1;
                int totweight = u.weight;

                // Tambahkan item selama masih cukup kapasitas
                while (j < items.size() && totweight + items[j].weight <= capacity) {
                    totweight += items[j].weight;
                    profit_bound += items[j].value;
                    j++;
                }

                // Tambahkan sebagian item terakhir jika masih ada ruang
                if (j < items.size())
                    profit_bound += (capacity - totweight) * ((double)items[j].value / items[j].weight);

                return profit_bound;
            };

            // Urutkan item berdasarkan rasio value/weight (greedy)
            std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
                return (double)a.value / a.weight > (double)b.value / b.weight;
            });

            std::priority_queue<Node, std::vector<Node>, CompareBound> Q;
            Q.emplace(-1, 0, 0, bound(Node(-1, 0, 0, 0))); // Masukkan root node

            int maxProfit = 0;
            while (!Q.empty()) {
                Node u = Q.top(); Q.pop();

                if (u.bound <= maxProfit || u.level == (int)items.size() - 1)
                    continue; // Skip jika bound-nya tidak menjanjikan atau sudah di level terakhir

                // Cabang: ambil item berikutnya
                Node v(u.level + 1,
                    u.profit + items[u.level + 1].value,
                    u.weight + items[u.level + 1].weight,
                    0);

                if (v.weight <= capacity && v.profit > maxProfit)
                    maxProfit = v.profit;

                v.bound = bound(v);
                if (v.bound > maxProfit) Q.push(v);

                // Cabang: tidak ambil item berikutnya
                v.weight = u.weight;
                v.profit = u.profit;
                v.bound = bound(v);
                if (v.bound > maxProfit) Q.push(v);
            }

            return maxProfit;
        }
        
        // Fungsi untuk menyelesaikan knapsack menggunakan FPTAS (Fully Polynomial Time Approximation Scheme)
        int solveFPTAS(double epsilon = 0.1) {
            int n = items.size();
            if (n == 0) return 0;

            // Temukan nilai maksimum dari semua item
            int maxVal = 0;
            for (const auto& item : items)
                maxVal = std::max(maxVal, item.value);

            // Hitung nilai skala berdasarkan epsilon
            double K = (epsilon * maxVal) / n;  // Skala untuk mendekatkan nilai
            if (K < 1e-9) K = 1e-9;             // Hindari pembagian dengan nol

            std::vector<int> scaled_values(n);  // Menyimpan nilai yang telah diskalakan
            int sum_scaled = 0;

            // Skala nilai agar kita bisa pakai DP berbasis nilai
            for (int i = 0; i < n; ++i) {
                scaled_values[i] = (int)(items[i].value / K);
                sum_scaled += scaled_values[i];
            }

            // DP[i][j] = minimum total weight to achieve scaled value j using first i items
            std::vector<std::vector<int>> dp(n + 1, std::vector<int>(sum_scaled + 1, INT_MAX));
            dp[0][0] = 0;  // Basis: tanpa item, value 0, weight 0

            // Dynamic Programming: pilih atau tidak pilih item
            for (int i = 1; i <= n; ++i) {
                int sv = scaled_values[i - 1]; // scaled value
                int w = items[i - 1].weight;
                for (int j = 0; j <= sum_scaled; ++j) {
                    dp[i][j] = dp[i - 1][j]; // Tidak memilih item ke-i
                    if (j >= sv && dp[i - 1][j - sv] != INT_MAX)
                        dp[i][j] = std::min(dp[i][j], dp[i - 1][j - sv] + w); // Pilih item ke-i
                }
            }

            // Cari nilai terbaik dengan berat <= kapasitas
            int best = 0;
            for (int j = 0; j <= sum_scaled; ++j)
                if (dp[n][j] <= capacity)
                    best = j;

            return (int)(best * K); // Konversi kembali ke nilai aslinya
        }

        // Greedy Local Search: menggabungkan greedy dengan local search untuk perbaikan
        int solveGreedyLocalSearch() {
            std::vector<int> selected(items.size(), 0);     // Menandai item yang diambil
            std::vector<int> indices(items.size());
            std::iota(indices.begin(), indices.end(), 0);   // Inisialisasi dengan 0..n-1

            // Urutkan berdasarkan rasio nilai/berat (greedy approach)
            std::sort(indices.begin(), indices.end(), [&](int a, int b) {
                return (double)items[a].value / items[a].weight > (double)items[b].value / items[b].weight;
            });

            int totalWeight = 0, totalValue = 0;

            // Ambil item sebanyak mungkin secara greedy
            for (int i : indices) {
                if (totalWeight + items[i].weight <= capacity) {
                    selected[i] = 1;
                    totalWeight += items[i].weight;
                    totalValue += items[i].value;
                }
            }

            // Local search: tukar satu item dengan item lain yang tidak dipilih untuk mencari solusi lebih baik
            bool improved = true;
            while (improved) {
                improved = false;
                for (size_t i = 0; i < items.size(); ++i) {
                    if (selected[i]) {
                        for (size_t j = 0; j < items.size(); ++j) {
                            if (!selected[j]) {
                                int newWeight = totalWeight - items[i].weight + items[j].weight;
                                int newValue  = totalValue - items[i].value + items[j].value;

                                // Tukar jika menghasilkan nilai lebih tinggi dan tidak melebihi kapasitas
                                if (newWeight <= capacity && newValue > totalValue) {
                                    selected[i] = 0;
                                    selected[j] = 1;
                                    totalWeight = newWeight;
                                    totalValue  = newValue;
                                    improved = true; // Ada perbaikan, lanjut iterasi
                                }
                            }
                        }
                    }
                }
            }

            return totalValue; // Nilai maksimum yang ditemukan
        }

        // Simulasi Deep Reinforcement Learning (DRL) untuk pengambilan keputusan
        // Catatan: Ini hanya simulasi, tidak ada model DRL yang sebenarnya
        int solveDRL() {        
            // Info: placeholder untuk metode DRL (Deep Reinforcement Learning)
            // std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulasi delay
            // std::cout << "\n[INFO] solveDRL: This method would load a trained DRL agent and run inference.\n";
            // std::cout << "[INFO] Simulating DRL decision-making...\n";

            // Inisialisasi indeks item 0..n-1
            std::vector<int> indices(items.size());
            std::iota(indices.begin(), indices.end(), 0); // isi: [0, 1, 2, ..., n-1]

            // Lakukan shuffle acak pada indeks (mensimulasikan pengambilan keputusan oleh agent DRL)
            std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});

            int totalWeight = 0, totalValue = 0;

            // Tambahkan item secara greedy dari urutan acak hingga kapasitas penuh
            for (int i : indices) {
                if (totalWeight + items[i].weight <= capacity) {
                    totalWeight += items[i].weight;
                    totalValue += items[i].value;
                }
            }

            return totalValue;
        }

    };

    // Kelas untuk knapsack fractional (Fractional Knapsack Problem)
    class FractionalKnapsack : public KnapsackBase {
    public:
        // Konstruktor menerima kapasitas dan meneruskan ke konstruktor induk
        FractionalKnapsack(int cap) : KnapsackBase(cap) {}

        // Mengembalikan jenis knapsack (untuk identifikasi)
        std::string type() const override { return "fractional"; }

        // Fungsi utama untuk menyelesaikan masalah knapsack menggunakan algoritma yang dipilih
        int solve(const std::string& algorithm = "greedy") override {
            if (algorithm == "greedy") return solveGreedy();                    // Pendekatan greedy klasik
            if (algorithm == "sort-then-fill") return solveSortThenFill();     // Versi greedy tanpa mengubah urutan asli items
            if (algorithm == "brute-force" && !DISABLE_BRUTE_FORCE) return solveBruteForce(0, capacity); // Brute-force rekursif (belum didefinisikan)
            if (algorithm == "brute-force-bitmask" && !DISABLE_BRUTE_FORCE) return solveBitmaskFraction(); // Bitmasking (belum didefinisikan)

            // Jika algoritma tidak dikenali
            std::cout << "Unsupported algorithm for fractional knapsack.\n";
            return -1;
        }

    private:
        // Algoritma greedy klasik untuk fractional knapsack
        int solveGreedy() {
            // Urutkan item berdasarkan rasio nilai terhadap berat secara menurun
            sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
                return a.ratio() > b.ratio();
            });

            double total = 0; // Menyimpan total nilai dari item yang dipilih
            int W = capacity; // Kapasitas tersisa dari knapsack

            // Iterasi setiap item dan ambil sebanyak mungkin dari item tersebut
            for (auto& i : items) {
                if (W >= i.weight) {
                    // Ambil seluruh item jika muat
                    total += i.value;
                    W -= i.weight;
                } else {
                    // Jika tidak cukup, ambil sebagian proporsional terhadap kapasitas tersisa
                    total += i.ratio() * W;
                    break; // Knapsack sudah penuh
                }
            }

            return (int)total; // Konversi ke bilangan bulat, dibulatkan ke bawah
        }

        // Versi alternatif dari greedy yang tidak memodifikasi urutan asli items
        int solveSortThenFill() {
            std::vector<Item> sortedItems = items; // Salin item agar tidak mengubah aslinya

            // Urutkan salinan berdasarkan rasio nilai terhadap berat secara menurun
            std::sort(sortedItems.begin(), sortedItems.end(), [](const Item& a, const Item& b) {
                return a.ratio() > b.ratio();
            });

            double total = 0;
            int W = capacity;

            // Isi knapsack sebanyak mungkin dengan item bernilai tinggi
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
        
        // Brute-force rekursif untuk fractional knapsack (mengambil sebagian item)
        double solveBruteForce(int idx, int remaining) {
            if (idx == items.size() || remaining == 0) return 0;

            // Jika berat item 0, abaikan untuk mencegah divide by zero
            if (items[idx].weight == 0) {
                return solveBruteForce(idx + 1, remaining);
            }

            double best = 0;
            int maxTake = std::min(items[idx].weight, remaining);

            for (int w = 0; w <= maxTake; ++w) {
                double takenValue = (double)items[idx].value * w / items[idx].weight;
                double rem = solveBruteForce(idx + 1, remaining - w);
                best = std::max(best, takenValue + rem);
            }

            return best;
        }

        // Fungsi untuk menyelesaikan fractional knapsack menggunakan bitmasking    
        int solveBitmaskFraction() {
            int n = items.size();
            double best = 0;
            int total = 1 << n; // Jumlah kombinasi 0/1 untuk n item

            for (int mask = 0; mask < total; ++mask) {
                int weight = 0;
                double value = 0;
                std::vector<Item> remainingItems;

                // Iterasi semua item untuk kombinasi 0/1
                for (int i = 0; i < n; ++i) {
                    if (mask & (1 << i)) {
                        // Jika dipilih secara penuh dan muat
                        if (weight + items[i].weight <= capacity) {
                            weight += items[i].weight;
                            value += items[i].value;
                        }
                    } else {
                        // Simpan item yang belum dipilih untuk kemungkinan pengambilan sebagian
                        remainingItems.push_back(items[i]);
                    }
                }

                int remCap = capacity - weight;

                // Jika masih ada kapasitas, coba ambil sebagian dari item sisa
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

    // Kelas untuk knapsack terbatas (Bounded Knapsack Problem)
    class BoundedKnapsack : public KnapsackBase {
    public:
        // Konstruktor: meneruskan nilai kapasitas ke KnapsackBase
        BoundedKnapsack(int cap) : KnapsackBase(cap) {}

        // Mengembalikan jenis knapsack sebagai "bounded"
        std::string type() const override { return "bounded"; }

        // Fungsi utama untuk menyelesaikan masalah knapsack berdasarkan nama algoritma yang diberikan
        int solve(const std::string& algorithm = "dp") override {
            if (algorithm == "brute-force" && !DISABLE_BRUTE_FORCE) return solveBruteForce(0, capacity);              // Eksplorasi semua kemungkinan jumlah item
            if (algorithm == "brute-force-bitmask" && !DISABLE_BRUTE_FORCE) return solveBruteForceBitmask();          // Cek semua subset dari item hasil ekspansi
            if (algorithm == "recursive-memo") return solveRecursiveMemo(0, capacity);        // Sama dengan brute-force tapi menggunakan memoization
            if (algorithm == "dp") return solveDP();                                          // Dynamic Programming klasik
            if (algorithm == "binary-split-dp") return solveBinarySplitDP();                  // Optimasi dengan binary-split
            if (algorithm == "branch-and-bound") return solveBranchAndBound();                // Pencarian terarah dengan pruning
            if (algorithm == "greedy-local") return solveGreedyLocalSearch();                 // Heuristik lokal

            std::cout << "Unsupported algorithm for Bounded Knapsack.\n";
            return -1;
        }

    private:    
        // Brute-force rekursif: coba semua kombinasi jumlah dari setiap item
        int solveBruteForce(int idx, int remaining) {
            if (idx == items.size()) return 0; // Basis rekursi: tidak ada item tersisa

            int maxVal = 0;

            // Coba semua jumlah item ke-idx dari 0 sampai maksimal quantity
            for (int q = 0; q <= items[idx].quantity; ++q) {
                int totalWeight = q * items[idx].weight;

                // Hanya pertimbangkan jika masih muat di kapasitas
                if (totalWeight <= remaining) {
                    int totalValue = q * items[idx].value;

                    // Rekursi ke item berikutnya dengan kapasitas yang dikurangi
                    maxVal = std::max(maxVal, totalValue +
                        solveBruteForce(idx + 1, remaining - totalWeight));
                }
            }

            return maxVal;
        }
        
        // Brute-force dengan pendekatan bitmask setelah memperluas item berdasarkan jumlah
        int solveBruteForceBitmask() {
            std::vector<Item> expanded;

            // Setiap item dengan quantity > 1 dipecah menjadi beberapa item 0/1
            for (auto& item : items) {
                for (int i = 0; i < item.quantity; ++i)
                    expanded.push_back(Item(item.value, item.weight));
            }

            int n = expanded.size();          // Jumlah total item setelah ekspansi
            int best = 0;
            int total = 1 << n;               // Total subset yang mungkin (2^n)

            // Iterasi setiap subset (kombinasi pemilihan item)
            for (int mask = 0; mask < total; ++mask) {
                int val = 0, wt = 0;

                // Periksa bit-bit untuk menentukan item mana yang dipilih
                for (int i = 0; i < n; ++i) {
                    if (mask & (1 << i)) {
                        val += expanded[i].value;
                        wt += expanded[i].weight;
                    }
                }

                // Jika total berat masih dalam kapasitas, simpan solusi terbaik
                if (wt <= capacity)
                    best = std::max(best, val);
            }

            return best;
        }
        
        // Memoisasi untuk menghindari perhitungan ulang submasalah
        std::map<std::pair<int, int>, int> memo;
        int solveRecursiveMemo(int idx, int remaining) {
            // Basis rekursi: jika semua item sudah diproses, kembalikan 0 (tidak ada nilai tambahan)
            if (idx == items.size()) return 0;

            // Gunakan pasangan (index item, sisa kapasitas) sebagai kunci memo
            auto key = std::make_pair(idx, remaining);

            // Jika sudah pernah dihitung, langsung kembalikan hasilnya
            if (memo.count(key)) return memo[key];

            int best = 0;

            // Coba semua kemungkinan jumlah pengambilan dari item ke-idx (0 sampai quantity)
            for (int q = 0; q <= items[idx].quantity; ++q) {
                int totalWeight = q * items[idx].weight;

                // Jika masih muat di kapasitas
                if (totalWeight <= remaining) {
                    int totalValue = q * items[idx].value;

                    // Panggil rekursif untuk item berikutnya, kapasitas berkurang
                    best = std::max(best, totalValue +
                        solveRecursiveMemo(idx + 1, remaining - totalWeight));
                }
            }

            // Simpan hasil dalam memo dan kembalikan
            return memo[key] = best;
        }

        // Dynamic Programming klasik untuk Bounded Knapsack
        int solveDP() {
            int n = items.size();

            // dp[i][w] menyimpan nilai maksimum menggunakan i item pertama dengan kapasitas w
            std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));

            // Iterasi semua item
            for (int i = 1; i <= n; ++i) {
                int val = items[i - 1].value;
                int wt = items[i - 1].weight;
                int qty = items[i - 1].quantity;

                // Iterasi semua kapasitas dari 0 sampai kapasitas maksimum
                for (int w = 0; w <= capacity; ++w) {
                    // Tidak ambil item i (nilai sama seperti sebelumnya)
                    dp[i][w] = dp[i - 1][w];

                    // Coba ambil 1 sampai qty dari item i jika masih muat
                    for (int q = 1; q <= qty; ++q) {
                        int totalWeight = q * wt;
                        if (totalWeight > w) break; // Tidak bisa ambil lebih
                        dp[i][w] = std::max(dp[i][w], dp[i - 1][w - totalWeight] + q * val);
                    }
                }
            }

            // Nilai maksimum untuk semua item dan kapasitas penuh
            return dp[n][capacity];
        }

        // Binary Split DP: menguraikan item dengan quantity > 1 menjadi beberapa item 0/1
        int solveBinarySplitDP() {
            std::vector<Item> splitItems;

            // Langkah 1: Pecah setiap item yang jumlahnya banyak menjadi beberapa item 0/1
            for (auto& item : items) {
                int q = item.quantity; // jumlah maksimum item yang boleh diambil
                int k = 1;             // faktor biner (1, 2, 4, 8, ...)
                while (q > 0) {
                    int take = std::min(k, q); // ambil sebanyak k atau sisa q jika lebih kecil

                    // Tambahkan item baru hasil penguraian (value dan weight dikalikan jumlah pengambilan)
                    splitItems.emplace_back(item.value * take, item.weight * take);

                    q -= take; // kurangi jumlah item yang tersisa
                    k *= 2;    // naikkan faktor biner
                }
            }

            // Langkah 2: Terapkan DP 1-dimensi pada hasil penguraian seperti knapsack 0/1
            std::vector<int> dp(capacity + 1, 0);

            // Iterasi seluruh item hasil split
            for (auto& item : splitItems) {
                // Lakukan iterasi dari kapasitas ke bawah untuk menghindari penggunaan ulang item
                for (int w = capacity; w >= item.weight; --w) {
                    // Update nilai maksimum yang bisa dicapai di kapasitas w
                    dp[w] = std::max(dp[w], dp[w - item.weight] + item.value);
                }
            }

            // Nilai maksimum pada kapasitas penuh
            return dp[capacity];
        }

        // Branch and Bound untuk Bounded Knapsack
        // Menggunakan struktur Node untuk menyimpan status solusi parsial
        int solveBranchAndBound() {
            // Struktur Node untuk menyimpan keadaan solusi parsial
            struct Node {
                int level, profit, weight;
                double bound;                 // Estimasi keuntungan maksimum dari node ini ke bawah
                std::vector<int> counts;      // Jumlah item yang telah dipilih per indeks

                Node(int l, int p, int w, double b, const std::vector<int>& c)
                    : level(l), profit(p), weight(w), bound(b), counts(c) {}
            };

            // Komparator untuk priority queue (max-heap berdasarkan bound terbesar)
            struct Compare {
                bool operator()(const Node& a, const Node& b) {
                    return a.bound < b.bound;  // bound tertinggi lebih diprioritaskan
                }
            };

            // Fungsi untuk menghitung upper bound dari node u
            auto bound = [&](const Node& u) {
                if (u.weight >= capacity) return 0.0;

                double profit_bound = u.profit;
                int weight = u.weight;

                // Coba ambil item mulai dari level berikutnya
                for (int i = u.level + 1; i < items.size(); ++i) {
                    int maxTake = std::min(items[i].quantity, (capacity - weight) / items[i].weight);
                    weight += maxTake * items[i].weight;
                    profit_bound += maxTake * items[i].value;

                    // Jika masih ada ruang, tambahkan fractional item untuk estimasi (relaksasi)
                    if (weight < capacity)
                        profit_bound += (capacity - weight) * ((double)items[i].value / items[i].weight);
                }

                return profit_bound;
            };

            // Priority queue untuk eksplorasi node secara best-first (bound tertinggi dulu)
            std::priority_queue<Node, std::vector<Node>, Compare> pq;
            std::vector<int> zero(items.size(), 0);  // Awal belum ambil item mana pun

            // Masukkan node awal (root)
            pq.emplace(-1, 0, 0, bound(Node(-1, 0, 0, 0.0, zero)), zero);

            int maxProfit = 0;

            // Proses antrian
            while (!pq.empty()) {
                Node u = pq.top(); pq.pop();

                // Jika sudah sampai akhir item, lewati node ini
                if (u.level == (int)items.size() - 1) continue;

                int next = u.level + 1;

                // Coba semua kemungkinan pengambilan item[next] dari 0 sampai quantity
                for (int k = 0; k <= items[next].quantity; ++k) {
                    int newWeight = u.weight + k * items[next].weight;
                    int newProfit = u.profit + k * items[next].value;

                    if (newWeight > capacity) break; // Lewati jika melebihi kapasitas

                    // Update pilihan item
                    std::vector<int> newCounts = u.counts;
                    newCounts[next] = k;

                    // Simpan solusi terbaik saat ini
                    if (newProfit > maxProfit)
                        maxProfit = newProfit;

                    // Hitung bound node baru
                    double bnd = bound(Node(next, newProfit, newWeight, 0.0, newCounts));

                    // Tambahkan node baru ke antrian jika promising
                    if (bnd > maxProfit) {
                        pq.emplace(next, newProfit, newWeight, bnd, newCounts);
                    }
                }
            }

            return maxProfit;
        }
        
        // Greedy Local Search: menggabungkan greedy dengan local search untuk perbaikan
        int solveGreedyLocalSearch() {
            // Membuat vektor indeks dari semua item (0, 1, ..., n-1)
            std::vector<int> indices(items.size());
            std::iota(indices.begin(), indices.end(), 0); // Mengisi vektor dengan nilai berturut-turut

            // Mengurutkan item berdasarkan rasio nilai/berat secara menurun (greedy choice)
            std::sort(indices.begin(), indices.end(), [&](int a, int b) {
                return (double)items[a].value / items[a].weight > (double)items[b].value / items[b].weight;
            });

            int totalWeight = 0, totalValue = 0;
            std::vector<int> taken(items.size(), 0); // Menyimpan jumlah item yang diambil dari setiap indeks

            // Ambil item sebanyak mungkin dari urutan yang telah diurutkan berdasarkan rasio
            for (int i : indices) {
                // Hitung berapa banyak item yang bisa diambil (tidak melebihi kuantitas dan kapasitas)
                int maxTake = std::min(items[i].quantity, (capacity - totalWeight) / items[i].weight);

                // Tambahkan ke total berat dan nilai
                totalWeight += maxTake * items[i].weight;
                totalValue += maxTake * items[i].value;

                // Simpan jumlah item yang diambil
                taken[i] = maxTake;
            }

            // Kembalikan total nilai dari item yang diambil
            return totalValue;
        }

    };

    // Kelas untuk knapsack tak terbatas (Unbounded Knapsack Problem)
    class UnboundedKnapsack : public KnapsackBase {
    public:
        using KnapsackBase::KnapsackBase;
        std::string type() const override { return "UnboundedKnapsack"; }

        int solve(const std::string& algorithm) override {
            if (algorithm == "brute-force" && !DISABLE_BRUTE_FORCE) return solveBruteForce(capacity);
            if (algorithm == "recursive-memo") return solveRecursiveMemo(capacity);
            if (algorithm == "dp") return solveDP();
            if (algorithm == "dp-1d") return solveDP1D();

            std::cout << "Unsupported algorithm for Unbounded Knapsack.\n";
            return -1;
        }

    private:    
        // Brute-force rekursif untuk Unbounded Knapsack
        int solveBruteForce(int remaining) {
            // Cek item dengan berat 0
            // for (const auto& item : items) {
            //     if (item.weight == 0 && item.value > 0) {
            //         std::cerr << "Error: Infinite recursion detected due to item with weight 0 and value > 0\n";
            //         return -1;
            //     }
            // }

            // // Debug awal setiap kelipatan 50
            // if (remaining % 50 == 0) {
            //     std::cout << "[DEBUG] Entering solveBruteForce(" << remaining << ")\n";
            // }

            // // Basis
            // if (remaining == 0) {
            //     std::cout << "[DEBUG] Reached base case with remaining = 0\n";
            //     return 0;
            // }

            // Basis: Jika tidak ada kapasitas tersisa, tidak bisa menambahkan item apa pun
            if (remaining == 0) return 0;

            int best = 0; // Nilai maksimum yang bisa dicapai

            // Coba semua item yang mungkin bisa dimasukkan
            for (const auto& item : items) {
                if (item.weight <= remaining) {
                    // int subResult = solveBruteForce(remaining - item.weight);
                    // Coba ambil item ini, dan lanjutkan sisa kapasitas
                    best = std::max(best, item.value + solveBruteForce(remaining - item.weight));

                    // Debug hanya saat remaining kecil agar tidak terlalu ramai
                    // if (remaining <= 5) {
                    //     std::cout << "[DEBUG] Trying item (val=" << item.value << ", w=" << item.weight 
                    //             << ") at remaining=" << remaining 
                    //             << " => subResult=" << subResult 
                    //             << ", current best=" << best << "\n";
                    // }
                }
            }

            // Kembalikan nilai maksimum yang bisa dicapai dari kapasitas ini
            return best;
        }

        // Memoisasi untuk menghindari perhitungan ulang submasalah
        std::unordered_map<int, int> memo;
        int solveRecursiveMemo(int remaining) {
            // Basis: tidak ada kapasitas tersisa
            if (remaining == 0) return 0;

            // Cek jika hasil sudah disimpan
            if (memo.count(remaining)) return memo[remaining];

            int best = 0;

            // Coba semua item yang masih bisa masuk ke kapasitas saat ini
            for (const auto& item : items) {
                if (item.weight <= remaining) {
                    best = std::max(best, item.value + solveRecursiveMemo(remaining - item.weight));
                }
            }

            // Simpan hasil untuk kapasitas ini agar tidak dihitung ulang
            return memo[remaining] = best;
        }
        
        // Dynamic Programming untuk Unbounded Knapsack
        int solveDP() {
            int n = items.size();  // Jumlah item yang tersedia

            // Matriks dp[i][w] menyimpan nilai maksimum yang dapat diperoleh
            // dengan menggunakan item pertama sampai ke-i, dan kapasitas w
            std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));

            // Iterasi untuk setiap item
            for (int i = 1; i <= n; ++i) {
                int wt = items[i - 1].weight;  // Berat item ke-(i-1)
                int val = items[i - 1].value;  // Nilai item ke-(i-1)

                // Iterasi untuk setiap kapasitas dari 0 hingga kapasitas maksimum
                for (int w = 0; w <= capacity; ++w) {
                    // Kasus tanpa mengambil item ke-i (lanjutkan solusi sebelumnya)
                    dp[i][w] = dp[i - 1][w];

                    // Kasus mengambil item ke-i setidaknya sekali
                    if (w >= wt)
                        // Karena ini *unbounded knapsack*, tetap di baris i (boleh ambil item ini lagi)
                        dp[i][w] = std::max(dp[i][w], dp[i][w - wt] + val);
                }
            }

            // Nilai maksimum yang dapat diperoleh dengan semua item dan kapasitas penuh
            return dp[n][capacity];
        }

        // Dynamic Programming satu dimensi untuk Unbounded Knapsack
        int solveDP1D() {
            // Inisialisasi array DP satu dimensi dengan ukuran (capacity + 1), semua nilai awal 0
            std::vector<int> dp(capacity + 1, 0);

            // Iterasi dari kapasitas 0 sampai kapasitas maksimum
            for (int w = 0; w <= capacity; ++w) {
                // Untuk setiap item, coba apakah bisa dimasukkan ke kapasitas saat ini
                for (const auto& item : items) {
                    if (item.weight <= w) {
                        // Jika item muat, update nilai maksimum di kapasitas w
                        // Perhatikan: dp[w - item.weight] + item.value karena item boleh diambil berulang
                        dp[w] = std::max(dp[w], dp[w - item.weight] + item.value);
                    }
                }
            }

            // Hasil akhir: nilai maksimum yang bisa dicapai dengan kapasitas penuh
            return dp[capacity];
        }

    };

    // Kelas dasar untuk knapsack multi-dimensi (Multi-Dimensional Knapsack Problem)
    class MultiDimensionalKnapsack : public KnapsackBase {
    public:
        std::vector<int> capacities;

        MultiDimensionalKnapsack(std::vector<int> caps, std::vector<Item> i)
            : KnapsackBase(caps[0]), capacities(std::move(caps)) {
            items = std::move(i);
        }

        std::string type() const override { return "MultiDimensionalKnapsack"; }

        int solve(const std::string& algorithm) override {
            if (algorithm == "brute-force" && !DISABLE_BRUTE_FORCE) return solveBruteForce(0, capacities);
            if (algorithm == "recursive-memo") return solveRecursiveMemo(0, capacities);
            if (algorithm == "dp") return solveDP();

            std::cout << "Unsupported algorithm for MultiDimensional Knapsack.\n";
            return -1;
        }

    private:    
        // Fungsi brute-force untuk menyelesaikan multidimensional 0/1 knapsack.
        // Memeriksa semua kemungkinan untuk memilih atau tidak memilih setiap item,
        // dengan mempertimbangkan beberapa kapasitas (dimensi) secara bersamaan.
        int solveBruteForce(int idx, std::vector<int> remaining) {
            // Basis rekursi: semua item telah diproses
            if (idx == items.size()) return 0;

            // Opsi 1: tidak ambil item ke-idx
            int without = solveBruteForce(idx + 1, remaining);

            // Periksa apakah item ke-idx dapat diambil (cukup kapasitas di semua dimensi)
            bool canTake = true;
            for (size_t d = 0; d < remaining.size(); ++d) {
                if (items[idx].weight > remaining[d]) {
                    canTake = false;
                    break;
                }
            }

            int with = 0;
            if (canTake) {
                // Opsi 2: ambil item ke-idx, lalu kurangi kapasitas di semua dimensi
                std::vector<int> updated = remaining;
                for (size_t d = 0; d < updated.size(); ++d)
                    updated[d] -= items[idx].weight;

                // Tambahkan nilai item dan lanjutkan rekursi
                with = items[idx].value + solveBruteForce(idx + 1, updated);
            }

            // Kembalikan nilai maksimum dari dua pilihan
            return std::max(without, with);
        }
        
        // Fungsi rekursif dengan memoization untuk menyelesaikan multidimensional 0/1 knapsack.
        // Menghindari perhitungan berulang dengan menyimpan hasil submasalah berdasarkan indeks item dan kapasitas tersisa pada setiap dimensi.
        std::map<std::tuple<int, std::vector<int>>, int> memo;
        int solveRecursiveMemo(int idx, std::vector<int> remaining) {
            // Basis: jika semua item sudah diproses, nilai total adalah 0
            if (idx == items.size()) return 0;

            // Gunakan tuple (idx, remaining) sebagai kunci cache
            auto key = std::make_tuple(idx, remaining);
            if (memo.count(key)) return memo[key];

            // Opsi 1: tidak ambil item ke-idx
            int without = solveRecursiveMemo(idx + 1, remaining);

            // Periksa apakah item ke-idx bisa diambil (cukup kapasitas di semua dimensi)
            bool canTake = true;
            for (size_t d = 0; d < remaining.size(); ++d) {
                if (items[idx].weight > remaining[d]) {
                    canTake = false;
                    break;
                }
            }

            int with = 0;
            if (canTake) {
                // Opsi 2: ambil item ke-idx, kurangi kapasitas pada semua dimensi
                std::vector<int> updated = remaining;
                for (size_t d = 0; d < updated.size(); ++d)
                    updated[d] -= items[idx].weight;

                // Tambahkan nilai dan lanjutkan ke item berikutnya
                with = items[idx].value + solveRecursiveMemo(idx + 1, updated);
            }

            // Simpan hasil dalam memo dan kembalikan nilai maksimum
            return memo[key] = std::max(without, with);
        }

        // Fungsi dynamic programming untuk menyelesaikan multidimensional 0/1 knapsack.
        // Mendukung 1D dan 2D knapsack secara eksplisit, dan memberikan peringatan jika jumlah dimensi lebih dari 2.
        int solveDP() {
            int n = items.size();         // Jumlah item
            int D = capacities.size();    // Jumlah dimensi (misal: 1 untuk berat saja, 2 untuk berat dan volume)

            // Jika jumlah dimensi adalah 2, gunakan DP 2D
            if (D == 2) {
                int C1 = capacities[0], C2 = capacities[1];
                std::vector<std::vector<int>> dp(C1 + 1, std::vector<int>(C2 + 1, 0));

                // Iterasi setiap item
                for (int k = 0; k < n; ++k) {
                    // Iterasi dari kapasitas maksimum ke bawah (agar tidak overwrite nilai sebelumnya)
                    for (int i = C1; i >= items[k].weight; --i) {
                        for (int j = C2; j >= items[k].weight; --j) {
                            // Update DP jika item ke-k diambil
                            dp[i][j] = std::max(dp[i][j], dp[i - items[k].weight][j - items[k].weight] + items[k].value);
                        }
                    }
                }
                return dp[C1][C2];  // Hasil akhir: nilai maksimum untuk kapasitas penuh di kedua dimensi
            }

            // Jika hanya satu dimensi, gunakan DP 1D biasa
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

            // Jika jumlah dimensi lebih dari 2, solusi DP belum diimplementasikan
            std::cout << "Too many dimensions for DP (>2).\n";
            return -1;
        }

    };

    // Kelas untuk knapsack multi-objektif (Multi-Objective Knapsack Problem)
    class MultiObjectiveKnapsack : public KnapsackBase {
    public:
        std::vector<int> secondaryValues;

        MultiObjectiveKnapsack(int c, std::vector<Item> i, std::vector<int> secVals)
            : KnapsackBase(c), secondaryValues(std::move(secVals)) {
            items = std::move(i);
        }

        std::string type() const override { return "MultiObjectiveKnapsack"; }

        int solve(const std::string& algorithm) override {
            if (algorithm == "brute-force" && DISABLE_BRUTE_FORCE) return solveBruteForce();
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

        // Fungsi pembanding dominasi antara dua solusi dalam konteks multi-objective knapsack.
        // Mengembalikan true jika solusi 'a' mendominasi solusi 'b',
        // yaitu jika 'a' setidaknya sebaik 'b' dalam semua tujuan dan lebih baik dalam setidaknya satu tujuan.
        bool dominates(const Solution& a, const Solution& b) {
            return a.primary >= b.primary && a.secondary >= b.secondary &&
                (a.primary > b.primary || a.secondary > b.secondary);
        }

        // Fungsi brute-force untuk menyelesaikan Multi-Objective 0/1 Knapsack Problem.
        // Mencari solusi terbaik berdasarkan nilai utama (primary), sambil mempertimbangkan nilai sekunder (secondary)
        // dan menghasilkan himpunan solusi pareto-optimal (non-dominated solutions).
        int solveBruteForce() {
            int n = items.size();               // Jumlah item yang tersedia
            if (secondaryValues.size() != static_cast<std::size_t>(n)) {
                return -1;
            }
            int bestPrimary = 0;                // Menyimpan nilai utama terbaik yang ditemukan
            std::vector<Solution> pareto;       // Menyimpan himpunan solusi pareto-optimal

            // Menjelajahi semua kemungkinan kombinasi item (2^n subset)
            for (int mask = 0; mask < (1 << n); ++mask) {
                int w = 0, v1 = 0, v2 = 0;      // Total berat, nilai utama, dan nilai sekunder untuk subset ini
                for (int i = 0; i < n; ++i) {
                    if (mask & (1 << i)) {
                        w += items[i].weight;
                        v1 += items[i].value;
                        v2 += secondaryValues[i]; // Menggunakan nilai sekunder dari vektor tambahan
                    }
                }

                // Hanya pertimbangkan subset yang muat dalam kapasitas
                if (w <= capacity) {
                    Solution sol = {w, v1, v2};     // Buat solusi baru dari subset saat ini
                    bool dominated = false;

                    // Periksa apakah solusi ini didominasi oleh solusi pareto yang sudah ada
                    for (const auto& p : pareto) {
                        if (dominates(p, sol)) {
                            dominated = true;
                            break;
                        }
                    }

                    // Jika tidak didominasi, tambahkan ke himpunan pareto dan update nilai terbaik
                    if (!dominated) {
                        pareto.push_back(sol);
                        bestPrimary = std::max(bestPrimary, v1);
                    }
                }
            }

            return bestPrimary; // Mengembalikan nilai utama tertinggi dari solusi pareto
        }
        
        // Fungsi brute-force untuk menyelesaikan Multi-Objective Knapsack dengan pendekatan leksikografis.
        // Memprioritaskan nilai utama (primary value), dan jika terdapat hasil yang sama, memilih yang nilai sekundernya lebih besar.
        int solveLexicographic() {
            int n = items.size();            // Jumlah item
            if (secondaryValues.size() != static_cast<std::size_t>(n)) {
                    return -1;
            }
            int bestPrimary = 0;             // Nilai utama terbaik
            int bestSecondary = 0;           // Nilai sekunder terbaik untuk nilai utama yang sama

            // Coba semua kemungkinan kombinasi subset dari item
            for (int mask = 0; mask < (1 << n); ++mask) {
                int w = 0, v1 = 0, v2 = 0;   // Total berat, nilai utama, dan nilai sekunder dari subset
                for (int i = 0; i < n; ++i) {
                    if (mask & (1 << i)) {
                        w += items[i].weight;
                        v1 += items[i].value;
                        v2 += secondaryValues[i];  // Mengambil nilai sekunder dari vektor eksternal
                    }
                }

                // Hanya pertimbangkan subset yang muat di dalam kapasitas
                if (w <= capacity) {
                    // Update solusi jika lebih baik secara leksikografis
                    if (v1 > bestPrimary || (v1 == bestPrimary && v2 > bestSecondary)) {
                        bestPrimary = v1;
                        bestSecondary = v2;
                    }
                }
            }

            return bestPrimary;  // Kembalikan nilai utama terbaik yang ditemukan
        }
        
        // Fungsi brute-force untuk menyelesaikan Multi-Objective Knapsack menggunakan metode Weighted Sum.
        // Nilai gabungan dihitung sebagai: score = primary + lambda * secondary,
        // kemudian memilih solusi dengan skor tertinggi (mengembalikan nilai utamanya saja).
        int solveWeightedSum(double lambda) {
            int n = items.size();               // Jumlah item
            if (secondaryValues.size() != static_cast<std::size_t>(n)) {
                    return -1;
            }
            double bestScore = 0;               // Skor gabungan terbaik
            int bestPrimary = 0;                // Menyimpan nilai utama dari solusi dengan skor terbaik

            // Jelajahi seluruh kemungkinan kombinasi item (2^n subset)
            for (int mask = 0; mask < (1 << n); ++mask) {
                int w = 0, v1 = 0, v2 = 0;      // Berat total, nilai utama, nilai sekunder
                for (int i = 0; i < n; ++i) {
                    if (mask & (1 << i)) {
                        w += items[i].weight;
                        v1 += items[i].value;
                        v2 += secondaryValues[i];
                    }
                }

                // Hanya pertimbangkan jika berat tidak melebihi kapasitas
                if (w <= capacity) {
                    double score = v1 + lambda * v2;  // Hitung skor gabungan
                    if (score > bestScore) {
                        bestScore = score;            // Perbarui skor terbaik jika lebih tinggi
                        bestPrimary = v1;             // Simpan nilai utama dari solusi tersebut
                    }
                }
            }

            return bestPrimary;  // Kembalikan nilai utama dari solusi terbaik menurut weighted sum
        }

        // Fungsi Dynamic Programming untuk menyelesaikan Multi-Objective 0/1 Knapsack Problem dengan pendekatan Pareto-optimal.
        // Menyimpan semua solusi non-dominated (tidak terdominasi) di setiap bobot dan memilih solusi dengan nilai utama terbaik.
        int solveParetoDP() {
            int n = items.size();                              // Jumlah item
            if (secondaryValues.size() != static_cast<std::size_t>(n)) {
                return -1; // Pastikan ukuran vektor nilai sekunder sesuai dengan jumlah item
            }
            std::vector<std::vector<Solution>> dp(capacity + 1); // dp[w] menyimpan solusi Pareto untuk kapasitas w
            dp[0].push_back({0, 0, 0});                        // Solusi dasar: kapasitas 0, nilai utama dan sekunder 0

            // Iterasi untuk setiap item
            for (int i = 0; i < n; ++i) {
                int wt = items[i].weight;                     // Berat item
                int v1 = items[i].value;                      // Nilai utama
                int v2 = secondaryValues[i];                  // Nilai sekunder (dari vektor eksternal)

                // Iterasi mundur dari kapasitas maksimum ke berat item
                for (int w = capacity; w >= wt; --w) {
                    // Untuk setiap solusi pada kapasitas w - wt
                    for (const auto& s : dp[w - wt]) {
                        // Buat solusi baru dengan memasukkan item i
                        Solution newSol = {w, s.primary + v1, s.secondary + v2};
                        bool dominated = false;

                        // Periksa apakah solusi ini didominasi oleh solusi lain pada kapasitas w
                        for (const auto& existing : dp[w]) {
                            if (dominates(existing, newSol)) {
                                dominated = true;
                                break;
                            }
                        }

                        // Jika tidak didominasi, tambahkan ke daftar solusi Pareto pada kapasitas w
                        if (!dominated) {
                            dp[w].push_back(newSol);
                        }
                    }
                }
            }

            // Ambil nilai utama terbaik dari semua solusi Pareto yang ada
            int best = 0;
            for (const auto& solutions : dp) {
                for (const auto& s : solutions) {
                    best = std::max(best, s.primary);
                }
            }

            return best; // Kembalikan nilai utama maksimum dari solusi Pareto
        }

    };

    // Kelas untuk knapsack dengan beberapa tas (Multiple Knapsack Problem)
    class MultipleKnapsack : public KnapsackBase {
    public:
        std::vector<int> bagCapacities;

        MultipleKnapsack(std::vector<int> caps, std::vector<Item> i)
            : KnapsackBase(0, i), bagCapacities(std::move(caps)) {}

        std::string type() const override { return "MultipleKnapsack"; }

        int solve(const std::string& algorithm) override {
            if (algorithm == "brute-force" && !DISABLE_BRUTE_FORCE) return solveBruteForce(0, bagCapacities, 0);
            if (algorithm == "greedy") return solveGreedy();
            if (algorithm == "dp-each-bag") return solveDPEachBag();

            std::cout << "Unsupported algorithm for MultipleKnapsack.\n";
            return -1;
        }

    private:    
        // Fungsi brute-force untuk menyelesaikan Multiple Knapsack Problem (MKP).
        // Menempatkan item ke salah satu knapsack yang tersedia (jika muat), atau melewatkannya.
        // Mengembalikan nilai maksimum yang dapat dicapai dengan semua kemungkinan distribusi.
        int solveBruteForce(int idx, std::vector<int> bags, int currentValue) {
            // Basis rekursi: jika semua item sudah diproses, kembalikan nilai saat ini
            if (idx == items.size()) return currentValue;

            int best = currentValue;

            // Coba tempatkan item idx ke setiap knapsack (bag) yang masih memiliki kapasitas cukup
            for (size_t b = 0; b < bags.size(); ++b) {
                if (bags[b] >= items[idx].weight) {
                    bags[b] -= items[idx].weight; // Kurangi kapasitas knapsack b
                    // Rekursi ke item berikutnya dengan nilai saat ini + nilai item yang diambil
                    best = std::max(best, solveBruteForce(idx + 1, bags, currentValue + items[idx].value));
                    bags[b] += items[idx].weight; // Kembalikan kapasitas knapsack b (backtracking)
                }
            }

            // Coba juga opsi untuk tidak memilih item idx sama sekali
            best = std::max(best, solveBruteForce(idx + 1, bags, currentValue));

            return best; // Kembalikan nilai maksimum dari semua kemungkinan
        }

        // Fungsi greedy untuk menyelesaikan Multiple Knapsack Problem (MKP).
        // Strategi: Urutkan item berdasarkan rasio nilai/berat secara menurun,
        // lalu tempatkan setiap item ke knapsack dengan kapasitas tersisa terkecil yang masih cukup menampung item.
        int solveGreedy() {
            std::vector<Item> sorted = items;
            
            // Urutkan item berdasarkan rasio nilai/berat (value-to-weight ratio) dari tinggi ke rendah
            std::sort(sorted.begin(), sorted.end(), [](const Item& a, const Item& b) {
                return a.ratio() > b.ratio();
            });

            std::vector<int> remaining = bagCapacities;  // Kapasitas tersisa di masing-masing knapsack
            int totalValue = 0;                          // Menyimpan total nilai dari item yang berhasil ditempatkan

            // Iterasi setiap item dari yang memiliki rasio terbaik
            for (const auto& item : sorted) {
                int bestBag = -1;           // Indeks knapsack terbaik yang akan dipilih
                int minSpace = INT_MAX;     // Ruang tersisa terkecil (untuk meminimalkan sisa ruang)

                // Temukan knapsack yang cukup untuk item ini dan memiliki ruang paling pas
                for (size_t i = 0; i < remaining.size(); ++i) {
                    if (remaining[i] >= item.weight && remaining[i] < minSpace) {
                        bestBag = i;
                        minSpace = remaining[i];
                    }
                }

                // Jika ditemukan knapsack yang cocok, tempatkan item
                if (bestBag != -1) {
                    remaining[bestBag] -= item.weight;
                    totalValue += item.value;
                }
            }

            return totalValue;  // Kembalikan total nilai dari solusi greedy
        }
        
        // Fungsi Dynamic Programming untuk menyelesaikan Multiple Knapsack Problem (MKP) dengan pendekatan satu per satu (each bag).
        // Menyelesaikan setiap knapsack secara independen menggunakan 0/1 Knapsack DP, memastikan setiap item hanya digunakan satu kali secara global.
        int solveDPEachBag() {
            int totalValue = 0;                            // Total nilai dari semua knapsack
            std::vector<bool> used(items.size(), false);  // Menandai item yang sudah digunakan oleh knapsack sebelumnya

            // Proses setiap knapsack satu per satu
            for (int cap : bagCapacities) {
                int n = items.size();
                // Matriks DP untuk knapsack saat ini (kapasitas = cap)
                std::vector<std::vector<int>> dp(n + 1, std::vector<int>(cap + 1, 0));

                // Isi DP seperti pada 0/1 Knapsack biasa
                for (int i = 1; i <= n; ++i) {
                    for (int w = 0; w <= cap; ++w) {
                        dp[i][w] = dp[i - 1][w];  // Tidak memilih item ke-i
                        if (!used[i - 1] && w >= items[i - 1].weight) {
                            // Jika item belum digunakan dan bisa dimasukkan, pilih yang maksimal
                            dp[i][w] = std::max(dp[i][w], dp[i - 1][w - items[i - 1].weight] + items[i - 1].value);
                        }
                    }
                }

                // Lacak kembali (backtrack) untuk mengetahui item mana yang dipilih
                int w = cap;
                for (int i = n; i >= 1; --i) {
                    if (dp[i][w] != dp[i - 1][w]) {
                        used[i - 1] = true;  // Tandai item sebagai telah digunakan
                        w -= items[i - 1].weight;
                        totalValue += items[i - 1].value;
                    }
                }
            }

            return totalValue;  // Kembalikan total nilai dari semua knapsack
        }

    };

    // Kelas untuk Quadratic Knapsack Problem (QKP)
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
            if (algorithm == "brute-force" && !DISABLE_BRUTE_FORCE) return solveBruteForce();
            if (algorithm == "greedy") return solveGreedyIgnoreQ();
            if (algorithm == "dp-approx") return solveDPIgnoreQ();

            std::cout << "Unsupported algorithm for QuadraticKnapsack.\n";
            return -1;
        }

        void setInteractionMatrix(const std::vector<std::vector<int>>& q) {
            Q = q;
        }

    private:    
        // Fungsi brute-force untuk menyelesaikan Quadratic Knapsack Problem (QKP).
        // Mengevaluasi semua subset dari item yang mungkin, lalu menghitung total nilai dan penalti interaksi antar item.
        // Fungsi ini mempertimbangkan: nilai individual item dan nilai tambahan dari pasangan item terpilih (matriks Q).
        int solveBruteForce() {
            int n = items.size(); // Jumlah item yang tersedia
            int best = 0;         // Menyimpan nilai maksimum yang ditemukan

            // Iterasi melalui semua kemungkinan subset item (2^n kemungkinan)
            for (int mask = 0; mask < (1 << n); ++mask) {
                int weight = 0;       // Total berat subset saat ini
                int value = 0;        // Nilai total dari subset
                std::vector<int> chosen; // Menyimpan indeks item yang dipilih dalam subset

                // Cek item mana saja yang disertakan dalam subset ini
                for (int i = 0; i < n; ++i) {
                    if (mask & (1 << i)) {
                        weight += items[i].weight; // Tambahkan berat item
                        value += items[i].value;   // Tambahkan nilai individual item
                        chosen.push_back(i);       // Simpan indeks item
                    }
                }

                // Lewati subset jika total berat melebihi kapasitas
                if (weight > capacity) continue;

                // Tambahkan nilai pasangan item dari matriks Q
                // Misalnya, jika item i dan j dipilih, tambahkan Q[i][j] ke total nilai
                for (size_t i = 0; i < chosen.size(); ++i)
                    for (size_t j = i + 1; j < chosen.size(); ++j)
                        value += Q[chosen[i]][chosen[j]];

                // Perbarui nilai terbaik jika solusi ini lebih baik
                best = std::max(best, value);
            }

            return best; // Kembalikan nilai maksimum dari semua subset valid
        }
        
        // Fungsi greedy untuk menyelesaikan Quadratic Knapsack Problem (QKP) dengan mengabaikan nilai Q[i][j] (interaksi antar item).
        // Strategi: memilih item berdasarkan rasio nilai/berat tertinggi (value-to-weight ratio), sampai kapasitas penuh atau tidak bisa lagi menambah item.
        int solveGreedyIgnoreQ() {
            std::vector<Item> sorted = items;

            // Urutkan item berdasarkan rasio nilai per berat secara menurun
            std::sort(sorted.begin(), sorted.end(), [](const Item& a, const Item& b) {
                return a.ratio() > b.ratio();  // Rasio = value / weight
            });

            int W = capacity;  // Kapasitas tersisa dari knapsack
            int value = 0;     // Total nilai yang dikumpulkan (tanpa mempertimbangkan Q[i][j])

            // Iterasi item yang telah diurutkan berdasarkan efisiensi rasio
            for (const auto& i : sorted) {
                if (W >= i.weight) {
                    W -= i.weight;     // Kurangi kapasitas jika item muat
                    value += i.value;  // Tambahkan nilai item
                }
            }

            return value;  // Kembalikan total nilai (dengan asumsi tidak ada efek interaksi antar item)
        }
        
        // Fungsi dynamic programming (DP) untuk menyelesaikan Quadratic Knapsack Problem (QKP) 
        // dengan mengabaikan kontribusi nilai interaksi Q[i][j].
        // Fungsi ini setara dengan penyelesaian 0/1 Knapsack klasik menggunakan DP 2D.
        int solveDPIgnoreQ() {
            int n = items.size();  // Jumlah item yang tersedia

            // Matriks DP: dp[i][w] menyimpan nilai maksimum dengan mempertimbangkan i item pertama dan kapasitas w
            std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));

            // Iterasi melalui setiap item
            for (int i = 1; i <= n; ++i) {
                int wt = items[i - 1].weight;  // Berat item ke-i
                int val = items[i - 1].value;  // Nilai item ke-i

                // Iterasi setiap kapasitas dari 0 hingga maksimum
                for (int w = 0; w <= capacity; ++w) {
                    if (wt > w) {
                        // Jika berat item melebihi kapasitas, tidak bisa diambil
                        dp[i][w] = dp[i - 1][w];
                    } else {
                        // Ambil maksimum antara tidak mengambil item ke-i dan mengambil item ke-i
                        dp[i][w] = std::max(dp[i - 1][w], dp[i - 1][w - wt] + val);
                    }
                }
            }

            return dp[n][capacity];  // Kembalikan nilai maksimum yang dapat diperoleh dengan kapasitas penuh
        }

    };

    // Kelas untuk Stochastic Knapsack Problem (SKP)
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
            return -1; // Default return for unsupported algorithms
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
        // Fungsi ini menggunakan pendekatan Monte Carlo untuk menyelesaikan Stochastic Knapsack Problem,
        // di mana bobot item tidak pasti, tetapi mengikuti distribusi probabilistik (weightProb).
        // Setiap percobaan (trial) akan menyampling bobot secara acak berdasarkan distribusi, 
        // lalu menyelesaikan 0/1 Knapsack menggunakan DP berdasarkan bobot yang disampling.
        int solveMonteCarlo(int trials) {
            int bestValue = 0;        // Menyimpan nilai terbaik yang ditemukan di antara semua percobaan
            int n = items.size();     // Jumlah item

            std::default_random_engine gen(std::random_device{}());  // Random engine untuk sampling

            for (int t = 0; t < trials; ++t) {
                std::vector<int> sampledWeights(n);  // Menyimpan bobot yang diambil dari distribusi probabilitas

                // Sampling bobot setiap item berdasarkan distribusi weightProb[i]
                for (int i = 0; i < n; ++i) {
                    double p = std::generate_canonical<double, 10>(gen);  // Nilai acak antara 0 dan 1
                    double sum = 0.0;

                    for (auto& wp : weightProb[i]) {
                        sum += wp.second;        // Akumulasi probabilitas
                        if (p <= sum) {          // Jika p berada dalam rentang probabilitas kumulatif
                            sampledWeights[i] = wp.first;  // Pilih bobot tersebut
                            break;
                        }
                    }
                }

                // Gunakan Dynamic Programming standar (0/1 Knapsack) dengan bobot hasil sampling
                std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));
                for (int i = 1; i <= n; ++i) {
                    for (int w = 0; w <= capacity; ++w) {
                        if (sampledWeights[i - 1] > w)
                            dp[i][w] = dp[i - 1][w];  // Item terlalu berat, tidak diambil
                        else
                            dp[i][w] = std::max(
                                dp[i - 1][w],
                                dp[i - 1][w - sampledWeights[i - 1]] + items[i - 1].value
                            );
                    }
                }

                // Simpan solusi terbaik dari percobaan ini
                bestValue = std::max(bestValue, dp[n][capacity]);
            }

            return bestValue;  // Kembalikan nilai maksimum dari seluruh percobaan
        }
        
        // Fungsi ini menyelesaikan versi stokastik dari knapsack problem menggunakan pendekatan greedy,
        // dengan memprioritaskan item berdasarkan rasio value / expected weight.
        // Bobot tiap item bersifat probabilistik dan dihitung nilai ekspektasinya.
        int solveGreedyExpected() {
            int n = items.size();  // Jumlah item
            std::vector<std::tuple<double, int>> sorted;  // Menyimpan (rasio nilai/ekspektasi bobot, indeks)

            // Hitung rasio nilai per bobot ekspektasi untuk setiap item
            for (int i = 0; i < n; ++i) {
                double expectedW = 0.0;  // Ekspektasi bobot untuk item i
                for (auto& [w, p] : weightProb[i]) 
                    expectedW += w * p;  // Rumus ekspektasi: E[W] = Σ(w * p)

                if (expectedW == 0) continue;  // Lewati item dengan ekspektasi bobot 0 (tidak valid)

                // Simpan rasio nilai per ekspektasi bobot beserta indeksnya
                sorted.emplace_back((double)items[i].value / expectedW, i);
            }

            // Urutkan item berdasarkan rasio nilai/ekspektasi bobot secara menurun (greedy choice)
            std::sort(sorted.begin(), sorted.end(), std::greater<>());

            int W = capacity;         // Kapasitas knapsack saat ini
            int totalValue = 0;       // Nilai total dari item yang diambil

            // Iterasi item dalam urutan rasio terbesar ke terkecil
            for (auto& [_, idx] : sorted) {
                double expectedW = 0;
                for (auto& [w, p] : weightProb[idx]) 
                    expectedW += w * p;

                int ew = (int)std::round(expectedW);  // Bulatkan ekspektasi bobot ke bilangan bulat

                // Ambil item jika masih muat ke dalam kapasitas
                if (W >= ew) {
                    W -= ew;
                    totalValue += items[idx].value;
                }
            }

            return totalValue;  // Kembalikan total nilai yang berhasil dimasukkan ke knapsack
        }
        
        // Fungsi ini menyelesaikan stochastic knapsack problem (item dengan bobot probabilistik) 
        // menggunakan pendekatan Dynamic Programming (DP) berdasarkan ekspektasi bobot (expected weight).
        int solveExpectedDP() {
            int n = items.size();  // Jumlah item
            std::vector<int> expectedWeights(n);  // Menyimpan bobot ekspektasi untuk setiap item

            // Hitung ekspektasi bobot untuk setiap item
            for (int i = 0; i < n; ++i) {
                double ew = 0.0;
                for (auto& [w, p] : weightProb[i]) 
                    ew += w * p;  // E[W] = Σ(w * p)

                // Bulatkan ke bilangan bulat karena DP tidak bisa pakai bobot pecahan
                expectedWeights[i] = (int)std::round(ew);
            }

            // Inisialisasi tabel DP: dp[i][w] = nilai maksimum dengan i item pertama dan kapasitas w
            std::vector<std::vector<int>> dp(n + 1, std::vector<int>(capacity + 1, 0));

            // Iterasi setiap item
            for (int i = 1; i <= n; ++i) {
                int wt = expectedWeights[i - 1];  // Bobot ekspektasi item ke-(i-1)
                int val = items[i - 1].value;     // Nilai item ke-(i-1)

                // Iterasi setiap kapasitas dari 0 hingga kapasitas maksimum
                for (int w = 0; w <= capacity; ++w) {
                    if (wt > w)
                        dp[i][w] = dp[i - 1][w];  // Tidak bisa ambil item karena bobot melebihi kapasitas
                    else
                        // Pilih maksimum antara tidak ambil atau ambil item
                        dp[i][w] = std::max(dp[i - 1][w], dp[i - 1][w - wt] + val);
                }
            }

            return dp[n][capacity];  // Nilai maksimum yang bisa dicapai dengan n item dan kapasitas penuh
        }

    };

    // Kelas untuk Multiple-Choice Knapsack Problem (MCKP)
    class MultiChoiceKnapsack : public KnapsackBase {
    public:
        std::vector<std::vector<Item>> groups;

        MultiChoiceKnapsack(int c, std::vector<std::vector<Item>> g)
            : KnapsackBase(c, {}), groups(std::move(g)) {}

        std::string type() const override { return "MultiChoiceKnapsack"; }

        int solve(const std::string& algorithm) override {
            if (algorithm == "brute-force" && !DISABLE_BRUTE_FORCE) return solveBruteForce(0, 0, 0);
            if (algorithm == "greedy") return solveGreedy();
            if (algorithm == "dp") return solveDP();

            std::cout << "Unsupported algorithm for MultiChoiceKnapsack.\n";
            return -1;
        }

    private:    
        // Fungsi ini menyelesaikan masalah Multiple-Choice Knapsack Problem (MCKP) 
        // menggunakan pendekatan brute-force rekursif.
        // Dalam MCKP, setiap item dibagi ke dalam beberapa grup, dan kita hanya boleh memilih maksimal satu item dari setiap grup.
        // Fungsi akan mencoba setiap kombinasi pemilihan satu item dari tiap grup dan mengembalikan nilai maksimum yang dapat dicapai
        // tanpa melebihi kapasitas total.
        int solveBruteForce(int groupIndex, int totalWeight, int totalValue) {
            // Basis: jika semua grup telah diproses
            if (groupIndex == groups.size()) {
                // Jika total berat tidak melebihi kapasitas, hasil valid dan kembalikan total value
                // Jika melebihi kapasitas, kembalikan 0 karena tidak valid
                return totalWeight <= capacity ? totalValue : 0;
            }

            int best = 0;  // Variabel untuk menyimpan hasil terbaik dari grup saat ini

            // Iterasi setiap item di grup saat ini
            for (const Item& item : groups[groupIndex]) {
                if (totalWeight + item.weight <= capacity) {
                    // Coba ambil item ini dan lanjut ke grup berikutnya
                    best = std::max(best, solveBruteForce(
                        groupIndex + 1,
                        totalWeight + item.weight,
                        totalValue + item.value
                    ));
                } else {
                    // Jika item tidak muat, tetap lanjut ke grup berikutnya tanpa memilih item ini
                    best = std::max(best, solveBruteForce(
                        groupIndex + 1,
                        totalWeight,
                        totalValue
                    ));
                }
            }

            return best;  // Kembalikan nilai terbaik dari semua pilihan yang mungkin
        }
        
        // Fungsi ini menyelesaikan masalah Multiple-Choice Knapsack Problem (MCKP) 
        // dengan pendekatan greedy (serakah).
        // Untuk setiap grup item, fungsi akan memilih item dengan rasio value/weight tertinggi,
        // selama item tersebut masih muat dalam kapasitas yang tersedia.
        // Fungsi ini cepat namun tidak menjamin solusi optimal.
        int solveGreedy() {
            int totalWeight = 0;  // Total berat item yang telah dipilih
            int totalValue = 0;   // Total nilai item yang telah dipilih

            // Iterasi setiap grup item
            for (auto& group : groups) {
                // Cari item terbaik dalam grup berdasarkan rasio nilai per berat (value / weight)
                auto bestIt = std::max_element(group.begin(), group.end(), [](const Item& a, const Item& b) {
                    return a.ratio() < b.ratio();  // Bandingkan rasio antar item
                });

                // Jika item terbaik dalam grup ini masih bisa dimasukkan ke knapsack tanpa melebihi kapasitas
                if (totalWeight + bestIt->weight <= capacity) {
                    totalWeight += bestIt->weight;  // Tambahkan berat item ke total
                    totalValue += bestIt->value;    // Tambahkan nilai item ke total
                }

                // Jika item terbaik dari grup tidak muat, grup tersebut dilewati (tidak memilih item dari grup tersebut)
            }

            return totalValue;  // Kembalikan total nilai dari item yang dipilih
        }

        
        // Fungsi ini menyelesaikan Multiple-Choice Knapsack Problem (MCKP)
        // menggunakan pendekatan Dynamic Programming (DP).
        // Setiap grup hanya boleh dipilih maksimal satu item, dan total berat tidak boleh melebihi kapasitas.
        // DP dilakukan per grup, sehingga hanya satu item dari setiap grup yang boleh dipertimbangkan.
        int solveDP() {
            int G = groups.size();  // Jumlah grup item
            // dp[g][w] menyimpan nilai maksimum jika mempertimbangkan grup 0..g-1 dan kapasitas w
            std::vector<std::vector<int>> dp(G + 1, std::vector<int>(capacity + 1, 0));

            // Iterasi setiap grup
            for (int g = 1; g <= G; ++g) {
                // Iterasi setiap kapasitas dari 0 sampai kapasitas total
                for (int w = 0; w <= capacity; ++w) {
                    // Awalnya, dp[g][w] diisi dengan solusi tanpa memilih item dari grup ke-g
                    dp[g][w] = dp[g - 1][w];

                    // Coba setiap item dalam grup ke-(g-1) (karena indeks array mulai dari 0)
                    for (const auto& item : groups[g - 1]) {
                        // Jika item muat dalam kapasitas w
                        if (w >= item.weight) {
                            // Pertimbangkan mengambil item ini dari grup dan bandingkan dengan yang sebelumnya
                            dp[g][w] = std::max(
                                dp[g][w],                          // tanpa memilih item ini
                                dp[g - 1][w - item.weight] + item.value  // jika memilih item ini
                            );
                        }
                    }
                }
            }

            // Nilai maksimum yang dapat dicapai dari semua grup dengan kapasitas total
            return dp[G][capacity];
        }

    };

    // Kelas untuk menyelesaikan masalah Knapsack menggunakan berbagai algoritma metaheuristik
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
        // Fungsi ini digunakan untuk mengevaluasi nilai total dari sebuah solusi knapsack,
        // berdasarkan array `selected` yang menunjukkan berapa banyak item ke-i yang dipilih.
        // Jika total berat melebihi kapasitas, maka solusi dianggap tidak valid dan nilai dikembalikan sebagai 0.

        static int evaluate(KnapsackBase* knapsack, const std::vector<int>& selected) {
            int totalWeight = 0;  // Menyimpan total berat dari item-item yang dipilih
            int totalValue = 0;   // Menyimpan total nilai dari item-item yang dipilih

            // Iterasi semua item dalam knapsack
            for (size_t i = 0; i < knapsack->items.size(); ++i) {
                // Tambahkan total berat dan nilai berdasarkan jumlah item yang dipilih (selected[i])
                totalWeight += knapsack->items[i].weight * selected[i];
                totalValue  += knapsack->items[i].value  * selected[i];
            }

            // Jika total berat tidak melebihi kapasitas, kembalikan nilai total; jika melebihi, solusi tidak valid → nilai 0
            return (totalWeight <= knapsack->capacity) ? totalValue : 0;
        }

        // Fungsi ini menyelesaikan masalah Knapsack menggunakan algoritma Simulated Annealing.
        // Tujuannya adalah untuk menemukan solusi mendekati optimal dengan menjelajahi tetangga dari solusi saat ini
        // dan menerima solusi yang lebih buruk dengan probabilitas tertentu (bergantung pada suhu) untuk menghindari local optimum.

        static int solveSimulatedAnnealing(KnapsackBase* knapsack) {
            int n = knapsack->items.size();  // Jumlah item dalam knapsack
            std::vector<int> current(n, 0),  // Solusi saat ini (vektor biner: 0 berarti tidak diambil, 1 berarti diambil)
                            best(n, 0);     // Solusi terbaik sejauh ini
            int bestVal = 0;                // Nilai terbaik sejauh ini

            double temp = 1000;             // Suhu awal (semakin tinggi, semakin besar kemungkinan menerima solusi buruk)
            double cooling = 0.95;          // Faktor pendinginan (turunkan suhu di setiap iterasi)
            std::mt19937 rng(std::random_device{}());  // Generator angka acak

            // Jalankan selama 1000 iterasi
            for (int iter = 0; iter < 1000; ++iter) {
                std::vector<int> neighbor = current;  // Salin solusi saat ini untuk dimodifikasi menjadi tetangga

                int i = rng() % n;                    // Pilih satu indeks item secara acak
                neighbor[i] = 1 - neighbor[i];        // Flip nilai item tersebut (ambil atau buang)

                int val = evaluate(knapsack, neighbor);  // Hitung nilai solusi tetangga

                // Jika solusi lebih baik, atau diterima secara probabilistik berdasarkan suhu
                if (val > bestVal || std::exp((val - bestVal) / temp) > (double)rng() / rng.max()) {
                    current = neighbor;               // Perbarui solusi saat ini
                    bestVal = val;                    // Perbarui nilai terbaik
                    best = neighbor;                  // Simpan solusi terbaik
                }

                temp *= cooling;  // Turunkan suhu
            }

            return bestVal;  // Kembalikan nilai terbaik yang ditemukan
        }

        // Fungsi ini menyelesaikan masalah Knapsack menggunakan algoritma Ant Colony Optimization (ACO).
        // Setiap "semut" membangun solusi berdasarkan probabilitas yang dipengaruhi oleh jejak feromon.
        // Setelah iterasi, feromon diperbarui untuk memperkuat jalur menuju solusi terbaik yang ditemukan.
        static int solveAntColony(KnapsackBase* knapsack) {
            int n = knapsack->items.size();                   // Jumlah item dalam knapsack
            std::vector<double> pheromone(n, 1.0);            // Inisialisasi feromon untuk setiap item
            int bestVal = 0;                                  // Menyimpan nilai terbaik sejauh ini
            std::vector<int> best;                            // Menyimpan solusi terbaik sejauh ini

            std::mt19937 rng(std::random_device{}());         // Random number generator

            // Lakukan iterasi sebanyak 50 kali (representasi dari jumlah generasi semut)
            for (int iter = 0; iter < 50; ++iter) {
                std::vector<int> candidate(n, 0);             // Solusi kandidat yang akan dibentuk oleh semut

                // Bangun solusi kandidat berdasarkan probabilitas feromon
                for (int i = 0; i < n; ++i)
                    candidate[i] = (rng() / (double)rng.max()) < pheromone[i] ? 1 : 0;

                int val = evaluate(knapsack, candidate);      // Evaluasi solusi kandidat

                // Jika kandidat lebih baik dari solusi terbaik sejauh ini, simpan
                if (val > bestVal) {
                    bestVal = val;
                    best = candidate;
                }

                // Perbarui feromon berdasarkan solusi kandidat (evaporasi dan deposisi)
                for (int i = 0; i < n; ++i)
                    pheromone[i] = 0.9 * pheromone[i] + 0.1 * candidate[i];
                // Feromon menguap sedikit (0.9), dan ditambah sedikit jika kandidat memilih item (0.1)
            }

            return bestVal; // Kembalikan nilai terbaik dari semua iterasi
        }

        // Fungsi ini menyelesaikan masalah Knapsack menggunakan algoritma Particle Swarm Optimization (PSO).
        // Dalam versi ini, hanya menggunakan satu partikel per iterasi yang dipilih secara acak (versi sederhana PSO).
        // Setiap partikel direpresentasikan sebagai vektor biner dan dievaluasi terhadap kapasitas knapsack.
        static int solveParticleSwarm(KnapsackBase* knapsack) {
            int n = knapsack->items.size();              // Jumlah item dalam knapsack
            std::vector<int> bestGlobal(n, 0);           // Solusi terbaik secara global
            int bestValue = 0;                           // Nilai terbaik global
            std::mt19937 rng(std::random_device{}());    // Random number generator

            // Iterasi sebanyak 100 kali untuk eksplorasi solusi (partikel)
            for (int iter = 0; iter < 100; ++iter) {
                std::vector<int> particle(n);            // Representasi partikel (solusi biner)

                // Inisialisasi partikel secara acak (0 atau 1)
                for (int i = 0; i < n; ++i)
                    particle[i] = rng() % 2;

                int val = evaluate(knapsack, particle);  // Evaluasi solusi partikel
                if (val > bestValue) {                   // Simpan solusi terbaik
                    bestValue = val;
                    bestGlobal = particle;
                }
            }

            return bestValue;                            // Kembalikan nilai solusi terbaik
        }

        // Fungsi ini merupakan placeholder untuk menyelesaikan Knapsack menggunakan Integer Linear Programming (ILP).
        // Dalam implementasi nyata, seharusnya menggunakan solver seperti CPLEX, Gurobi, atau OR-Tools.
        // Saat ini, sebagai simulasi, digunakan algoritma Simulated Annealing sebagai pendekatan alternatif.
        static int solveILP(KnapsackBase* knapsack) {
            // std::cout << "[ILP Placeholder] Ideally solved using CPLEX, Gurobi, or OR-Tools.\n";
            return solveSimulatedAnnealing(knapsack); // Menggunakan metode alternatif
        }

        // Fungsi ini merupakan placeholder untuk menyelesaikan Knapsack menggunakan Constraint Programming (CP).
        // Dalam implementasi sebenarnya, perlu digunakan library seperti Google OR-Tools yang mendukung constraint solver.
        // Saat ini, fungsi ini menggunakan algoritma Ant Colony Optimization sebagai pendekatan alternatif.
        static int solveCP(KnapsackBase* knapsack) {
            // std::cout << "[Constraint Programming Placeholder] Use Google OR-Tools or similar.\n";
            return solveAntColony(knapsack); // Menggunakan metode alternatif
        }

    };

    } // namespace knapsack 

    // Flag atomic untuk mengontrol spinner loading
    // Atomic agar aman diakses dari multiple thread
    std::atomic<bool> loadingDone(false);
    std::atomic<bool> interrupted(false);

    void handleInterrupt(int signal) {
        if (signal == SIGINT) {
            interrupted = true;
            std::cout << "\n[DEBUG] Ctrl+C detected. Stopping...\n";
        }
    }

    // Fungsi menampilkan loading spinner di terminal selama proses berlangsung
    // Parameter:
    // - message: teks yang tampil sebelum spinner (default "Processing")
    // Spinner animasi sederhana bergantian karakter | / - \
    // Loop berhenti jika loadingDone = true dan spinner sudah tampil minimal sekali
    // void ShowLoadingSpinner(const std::string& message = "Processing") {
    //     const char spinner[] = {'|', '/', '-', '\\'};  // Karakter animasi spinner
    //     int idx = 0;           // Indeks untuk karakter spinner saat ini
    //     bool printedOnce = false; // Flag untuk memastikan spinner tampil minimal sekali

    //     // Loop sampai loadingDone = true dan spinner sudah tampil sekali
    //     while (!loadingDone || !printedOnce) {
    //         // \r untuk kembalikan cursor ke awal baris agar spinner overwrite
    //         std::cout << "\r" << message << " " << spinner[idx++] << std::flush;
    //         printedOnce = true;
    //         if (idx == 4) idx = 0;   // Ulangi animasi spinner
    //         std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Delay 100ms antar frame
    //     }
    //     std::cout << "\r\n"; // Pindah baris setelah selesai
    // }

    // pake ANSI
    // void ShowLoadingSpinner(const std::string& message = "Processing") {
    //     const char spinner[] = {'|', '/', '-', '\\'};
    //     int idx = 0;
    //     bool printedOnce = false;

    //     while (!loadingDone || !printedOnce) {
    //         printedOnce = true;

    //         // Move cursor to beginning (\r), clear line (\033[2K)
    //         std::cout << "\r\033[2K" << message << " " << spinner[idx++] << std::flush;

    //         if (idx == 4) idx = 0;
    //         std::this_thread::sleep_for(std::chrono::milliseconds(100));
    //     }

    //     // Final clear to erase spinner
    //     std::cout << "\r\033[2K" << std::flush;
    // }

    const int totalIteration = 
                (MAX_NUMBER_ITEMS_TESTED / STEP_NUMBER_ITEMS) * 
                (MAX_CAPACITY_TESTED / STEP_CAPACITY) * 
                (MAX_QUANTITY_TESTED / STEP_QUANTITY) * 
                (MAX_ITEMS_VALUE_TESTED / STEP_ITEMS_VALUE) * 
                (MAX_ITEMS_WEIGHT_TESTED / STEP_ITEMS_WEIGHT);
    double progressPercentage = 0;

    void ShowLoadingSpinner(const std::string& message) {
        const char spinner[] = {'|', '/', '-', '\\'};
        int idx = 0;

        std::string lastOutput;

        while (!loadingDone && !interrupted) {
            std::ostringstream oss;
            oss << "\r" // Kembali ke awal baris
                << message
                << " " << spinner[idx]
                << " ["
                << std::fixed << std::setprecision(2)
                << std::setw(6) << progressPercentage
                << "%]";

            std::string currentOutput = oss.str();

            // Tambahkan spasi jika output sebelumnya lebih panjang, agar benar-benar tertimpa
            if (lastOutput.length() > currentOutput.length()) {
                currentOutput += std::string(lastOutput.length() - currentOutput.length(), ' ');
            }

            std::cout << currentOutput << std::flush;
            lastOutput = currentOutput;

            idx = (idx + 1) % 4;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        std::ostringstream doneMsg;
        doneMsg << "\r" << message << " Done. [100.00%]";
        if (lastOutput.length() > doneMsg.str().length()) {
            doneMsg << std::string(lastOutput.length() - doneMsg.str().length(), ' ');
        }
        std::cout << doneMsg.str() << std::endl;
    }


    // Fungsi untuk memformat angka dengan koma ribuan (misal 1000000 jadi 1,000,000)
    // Parameter: value angka bertipe long long
    // Return: string yang sudah diberi koma ribuan
    std::string FormatWithCommas(long long value) {
        std::string s = std::to_string(value);                    // Ubah angka jadi string
        int insert_position = static_cast<int>(s.length()) - 3;   // Posisi mulai sisipkan koma dari kanan
        while (insert_position > 0) {
            s.insert(insert_position, ",");                       // Sisipkan koma di posisi yang ditentukan
            insert_position -= 3;                                 // Geser 3 posisi ke kiri
        }
        return s;                                                 // Kembalikan string hasil format
    }

    // Function to generate random items
    std::vector<knapsack::Item> generateRandomItems(int numItems, int maxValue, int maxWeight, int maxQuantity = 1) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> valueDist(1, maxValue);
        std::uniform_int_distribution<> weightDist(1, maxWeight);
        std::uniform_int_distribution<> quantityDist(1, maxQuantity);

        std::vector<knapsack::Item> items;
        for (int i = 0; i < numItems; ++i) {
            int value = valueDist(gen);
            int weight = weightDist(gen);
            int quantity = maxQuantity > 1 ? quantityDist(gen) : 1;
            items.emplace_back(value, weight, quantity);
        }
        return items;
    }

    // Function to display items
    void displayItems(const std::vector<knapsack::Item>& items) {
        std::cout << "Items (value, weight, quantity):\n";
        for (size_t i = 0; i < items.size(); ++i) {
            std::cout << i + 1 << ": (" << items[i].value << ", " << items[i].weight 
                << ", " << items[i].quantity << ")\n";
        }
    }

    // Function to test a specific knapsack type and algorithm
    // void testKnapsack(knapsack::KnapsackBase* knapsack, const std::string& algorithm) {
    //     loadingDone = false;  // Reset flag loading sebelum mulai
    //     // Jalankan spinner di thread terpisah agar UI tetap responsif
    //     std::thread spinnerThread(ShowLoadingSpinner, ("\nTesting " + knapsack->type() + " with algorithm: " + algorithm));    
        
    //     auto start = std::chrono::high_resolution_clock::now();
    //     int result = knapsack->solve(algorithm);
    //     auto end = std::chrono::high_resolution_clock::now();

    //     // Beri tanda spinner berhenti
    //     loadingDone = true;
        
    //     // Tunggu thread spinner selesai
    //     spinnerThread.join();

    //     // Hitung durasi dalam nanodetik
    //     long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
    //     // Tampilkan hasil        
    //     std::cout << "Result: " << result << std::endl;
    //     std::cout << "Time taken: " << std::fixed << std::setprecision(6) << FormatWithCommas(duration) << " ns\n";
    // }

    // void testKnapsack(knapsack::KnapsackBase* knapsack, const std::string& algorithm) {
    //     loadingDone = false;
    //     std::thread spinnerThread(ShowLoadingSpinner, "Testing " + knapsack->type() + " with algorithm: " + algorithm);
        
    //     // Jangan panggil solve dulu, hanya simulasi
    //     std::this_thread::sleep_for(std::chrono::seconds(3));
    //     int result = 1234;  // dummy result

    //     loadingDone = true;
    //     spinnerThread.join();

    //     std::cout << "Result: " << result << std::endl;
    // }

    // void testKnapsack(knapsack::KnapsackBase* knapsack, const std::string& algorithm) {
    //     loadingDone = false;

    //     // Simpan buffer asli
    //     std::streambuf* orig_buf = std::cout.rdbuf();
    //     std::ostringstream temp_out;
    //     std::cout.rdbuf(temp_out.rdbuf()); // redirect ke buffer sementara

    //     std::thread spinnerThread(ShowLoadingSpinner, "Testing " + knapsack->type() + " with algorithm: " + algorithm);
        
    //     auto start = std::chrono::high_resolution_clock::now();
    //     int result = knapsack->solve(algorithm);
    //     auto end = std::chrono::high_resolution_clock::now();

    //     loadingDone = true;
    //     spinnerThread.join();

    //     std::cout.rdbuf(orig_buf); // kembalikan output normal

    //     long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    //     std::cout << "\nResult: " << result << std::endl;
    //     std::cout << "Time taken: " << std::fixed << std::setprecision(6) << FormatWithCommas(duration) << " ns\n";

    //     // (opsional) tampilkan isi output yang disimpan (buat lihat output dari solve)
    //     // std::cout << "Debug log:\n" << temp_out.str();
    // }

    void testKnapsack(knapsack::KnapsackBase* knapsack, 
                    const std::string& algorithm, 
                    const bool& saveToFile = false,
                    std::ofstream* out = nullptr,
                    const bool isMetaheuristic = false) {        

        std::string KnapsackType = isMetaheuristic ? "Metaheuristic" : knapsack->type();
        
        if (!saveToFile) {        

            loadingDone = false;
            int result;
            
            // Start spinner with the test message
            std::thread spinnerThread(ShowLoadingSpinner, 
                "Testing " + KnapsackType + " with algorithm: " + algorithm + " ");
            
            auto start = std::chrono::high_resolution_clock::now();
            if (isMetaheuristic) {
                // Jika ini adalah metaheuristik, panggil solver khusus
                result = knapsack::MetaheuristicKnapsackSolver::solve(knapsack, algorithm);
            } else {
                // Jika bukan metaheuristik, panggil solve biasa
                result = knapsack->solve(algorithm);
            }        
            auto end = std::chrono::high_resolution_clock::now();
            
            loadingDone = true;
            spinnerThread.join();
            
            // Now print the results on a new line
        
            long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            std::cout << "Result: " << result << std::endl;
            std::cout << "Time taken: " << std::fixed << std::setprecision(6) 
                    << FormatWithCommas(duration) << " ns\n\n";
            return;
        }

        // Jika ingin simpan ke file
        else if (saveToFile && out && out->is_open()) {

            loadingDone = false;
            progressPercentage = 0.0;
            int result;
            
            // Start spinner with the test message
            std::thread spinnerThread(ShowLoadingSpinner, 
                "Saving " + knapsack->type() + " with algorithm: " + algorithm + " ");    

            for (int id_item = 1; id_item <= MAX_NUMBER_ITEMS_TESTED; id_item += STEP_NUMBER_ITEMS) {
                for (int id_capacity = 1; id_capacity <= MAX_CAPACITY_TESTED; id_capacity += STEP_CAPACITY) {
                    for (int id_quantity = 1; id_quantity <= MAX_QUANTITY_TESTED; id_quantity += STEP_QUANTITY) {
                        for (int id_value = 1; id_value <= MAX_ITEMS_VALUE_TESTED; id_value += STEP_ITEMS_VALUE) {
                            for (int id_weight = 1; id_weight <= MAX_ITEMS_WEIGHT_TESTED; id_weight += STEP_ITEMS_WEIGHT) {

                                if (interrupted) {
                                    std::cout << "\n[DEBUG] Stopping due to Ctrl+C interrupt.\n";
                                    exit(1); // Stop if interrupted
                                }
                                
                                // kombinasi
                                int numItems = id_item;
                                int maxValue = id_value;
                                int maxWeight = id_weight;
                                int maxQuantity = id_quantity;
                                int capacity = id_capacity;

                                // Set kapasitas knapsack
                                knapsack->capacity = capacity;

                                // Generate items secara acak
                                knapsack->items = generateRandomItems(numItems, maxValue, maxWeight, maxQuantity);                                                        

                                // Jalankan algoritma yang diinginkan
                                auto start = std::chrono::high_resolution_clock::now();
                                if (isMetaheuristic) {
                                    // Jika ini adalah metaheuristik, panggil solver khusus
                                    result = knapsack::MetaheuristicKnapsackSolver::solve(knapsack, algorithm);
                                } else {
                                    // Jika bukan metaheuristik, panggil solve biasa
                                    result = knapsack->solve(algorithm);
                                }                                
                                auto end = std::chrono::high_resolution_clock::now();

                                long long duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

                                // Tulis hasil ke file
                                if (out) {
                                    (*out) << algorithm << "," 
                                        << numItems << "," << maxValue << "," 
                                        << maxWeight << "," << maxQuantity << "," 
                                        << capacity << "," << duration << ","
                                        << result << "\n";
                                }

                                progressPercentage += (100.0 / static_cast<double>(totalIteration));

                            }
                        }
                    }
                }
            }
            loadingDone = true;
            spinnerThread.join();
        }
        else if (saveToFile && (!out || !out->is_open())) {
            std::cerr << "\nError: Output file stream is not open.";
            return;
        }
    }

    void PrintBoxedText(const std::string& text) {
        const int padding = 5; // Spasi di kiri dan kanan teks
        int boxWidth = text.size() + padding * 2;

        // Top border
        std::cout << "\n\n+" << std::string(boxWidth, '-') << "+\n";

        // Middle line with text centered
        std::cout << "|" << std::string(padding, ' ') << text << std::string(padding, ' ') << "|\n";

        // Bottom border
        std::cout << "+" << std::string(boxWidth, '-') << "+\n";
    }

    // Main menu function
    void mainMenu() {
        int capacity;
        int numItems;
        int maxValue = 100;
        int maxWeight = 50;
        int maxQuantity = 5;
        std::vector<knapsack::Item> items;
        std::vector<int> capacities;
        std::vector<std::vector<int>> weightProb;
        std::vector<int> secondaryValues;
        std::vector<std::vector<knapsack::Item>> groups;
        std::vector<std::vector<int>> Q;

        std::cout << "Knapsack Problem Solver\n";
        std::cout << "Choose input method:\n";
        std::cout << "1. Random generation\n";
        std::cout << "2. Manual input\n";
        std::cout << "3. Save to file\n";
        std::cout << "4. Save all\n";
        std::cout << "5. Exit\n";
        std::cout << "Enter Choice (1 or 5): ";
        int inputChoice;
        std::cin >> inputChoice;

        bool saveToFile = false;
        if (inputChoice == 1 || inputChoice == 3) {
            if (inputChoice == 1) {
                std::cout << "Enter problem size (number of items): ";
                std::cin >> numItems;
                std::cout << "Enter maximum value for items: ";
                std::cin >> maxValue;
                std::cout << "Enter maximum weight for items: ";
                std::cin >> maxWeight;
                items = generateRandomItems(numItems, maxValue, maxWeight, maxQuantity);
            }        
            if (inputChoice == 3) { saveToFile = true; }
            else { displayItems(items); }
        } else if (inputChoice == 2) {
            std::cout << "Enter number of items: ";
            std::cin >> numItems;
            items.resize(numItems);
            for (int i = 0; i < numItems; ++i) {
                std::cout << "Item " << i + 1 << " - enter value, weight, and quantity: ";
                std::cin >> items[i].value >> items[i].weight >> items[i].quantity;
            }
        } else if (inputChoice == 4) {
            saveToFile = true;
            std::cout << "Saving all results to files.\n";
            std::cout << "This will take a while, please wait...\n";
        } 
        else if (inputChoice == 5) {
            std::cout << "Exiting program.\n";
            return; // Exit if user chooses to exit
        }
        else {
            std::cout << "Exiting program.\n";
            return; // Exit if user chooses to exit
        }
        
        int index = 0;
        do {
            int choice;
            if (inputChoice == 4) choice = index;
            else {
                std::cout << "\nSelect knapsack type:\n";
                std::cout << "1. 0/1 Knapsack\n";
                std::cout << "2. Fractional Knapsack\n";
                std::cout << "3. Bounded Knapsack\n";
                std::cout << "4. Unbounded Knapsack\n";
                std::cout << "5. Multi-Dimensional Knapsack\n";
                std::cout << "6. Multi-Objective Knapsack\n";
                std::cout << "7. Multiple Knapsack\n";
                std::cout << "8. Quadratic Knapsack\n";
                std::cout << "9. Stochastic Knapsack\n";
                std::cout << "10. Multiple-Choice Knapsack\n";
                std::cout << "11. Metaheuristic Approaches\n";
                std::cout << "Enter your choice (1-11): ";    
                std::cin >> choice;    
            }

            switch (choice) {
                case 1: { // 0/1 Knapsack
                    if (!saveToFile) {
                        std::cout << "Enter knapsack capacity: ";
                        std::cin >> capacity;
                        PrintBoxedText("0/1 Knapsack Problem");
                    } else {
                        capacity = DEFAULT_CAPACITY_TESTED; // nilai default saat saveToFile (disesuaikan jika perlu)
                        PrintBoxedText(std::string("Saving result to ") + OUTPUT_ZERO_ONE_KNAPSACK);
                    }

                    knapsack::ZeroOneKnapsack ks(capacity, items);

                    std::vector<std::string> algorithms = {
                        "dp", "dp-1d", "recursive-memo", "meet-in-the-middle",
                        "branch-and-bound", "fptas", "greedy-local"
                    };

                    if (!DISABLE_BRUTE_FORCE) {
                        algorithms.push_back("brute-force");
                        algorithms.push_back("brute-force-bitmask");
                    }

                    std::ofstream log;
                    if (saveToFile) {
                        std::string folder = "output";
                        if (!std::filesystem::exists(folder)) {
                            std::filesystem::create_directories(folder);
                        }   
                        std::string filename = folder + "/" + OUTPUT_ZERO_ONE_KNAPSACK;
                        log.open(filename);
                        if (!log.is_open()) {
                            std::cerr << "Failed to open file: " << OUTPUT_ZERO_ONE_KNAPSACK << "\n";
                            break; // hentikan eksekusi jika file tidak bisa dibuka
                        }
                        log << "Algorithm,TotalItems,ItemsValue,ItemsWeight,Quantity,Capacity,Time_ns,Result\n";
                    }

                    for (const auto& algo : algorithms) {
                        testKnapsack(&ks, algo, saveToFile, saveToFile ? &log : nullptr);
                    }

                    if (saveToFile) {
                        log.close();
                    }

                    break;
                }

                case 2: { // Fractional Knapsack
                    if (!saveToFile) {
                        std::cout << "Enter knapsack capacity: ";
                        std::cin >> capacity;
                        PrintBoxedText("Fractional Knapsack Problem");
                    } else {
                        capacity = DEFAULT_CAPACITY_TESTED;
                        PrintBoxedText(std::string("Saving result to ") + OUTPUT_FRACTIONAL_KNAPSACK);
                    }

                    knapsack::FractionalKnapsack ks(capacity);
                    ks.items = items;

                    std::vector<std::string> algorithms = {
                        "greedy", "sort-then-fill"
                    };

                    if (!DISABLE_BRUTE_FORCE) {
                        algorithms.push_back("brute-force");
                        algorithms.push_back("brute-force-bitmask");
                    }

                    std::ofstream log;
                    if (saveToFile) {
                        std::string folder = "output";
                        if (!std::filesystem::exists(folder)) {
                            std::filesystem::create_directories(folder);
                        }   
                        std::string filename = folder + "/" + OUTPUT_FRACTIONAL_KNAPSACK;
                        log.open(filename);
                        if (!log.is_open()) {
                            std::cerr << "Failed to open file: " << OUTPUT_FRACTIONAL_KNAPSACK << "\n";
                            break;
                        }
                        log << "Algorithm,TotalItems,ItemsValue,ItemsWeight,Quantity,Capacity,Time_ns,Result\n";
                    }

                    for (const auto& algo : algorithms) {
                        testKnapsack(&ks, algo, saveToFile, saveToFile ? &log : nullptr);
                    }

                    if (saveToFile) {
                        log.close();
                    }

                    break;
                }

                case 3: { // Bounded Knapsack
                    if (!saveToFile) {
                        std::cout << "Enter knapsack capacity: ";
                        std::cin >> capacity;
                        if (inputChoice == 1) {
                            std::cout << "Enter maximum quantity for each item: ";
                            std::cin >> maxQuantity;
                            items = generateRandomItems(numItems, maxValue, maxWeight, maxQuantity);
                        }
                        PrintBoxedText("Bounded Knapsack Problem");
                    } else {
                        capacity = DEFAULT_CAPACITY_TESTED; // Gunakan konstanta default saat saveToFile
                        PrintBoxedText("Saving result to " + std::string(OUTPUT_BOUNDED_KNAPSACK));
                    }

                    knapsack::BoundedKnapsack ks(capacity);
                    ks.items = items;

                    std::vector<std::string> algorithms = {
                        "dp", "binary-split-dp", "recursive-memo",                      
                        "branch-and-bound", "greedy-local"
                    };

                    if (!DISABLE_BRUTE_FORCE) {
                        algorithms.push_back("brute-force");
                        algorithms.push_back("brute-force-bitmask");
                    }

                    std::ofstream log;
                    if (saveToFile) {
                        std::string folder = "output";
                        if (!std::filesystem::exists(folder)) {
                            std::filesystem::create_directories(folder);
                        }   
                        std::string filename = folder + "/" + OUTPUT_BOUNDED_KNAPSACK;
                        log.open(filename);
                        if (!log.is_open()) {
                            std::cerr << "Error: Failed to open output file: " << OUTPUT_BOUNDED_KNAPSACK << "\n";
                            break;
                        }
                        log << "Algorithm,TotalItems,ItemsValue,ItemsWeight,Quantity,Capacity,Time_ns,Result\n";
                    }

                    for (const auto& algo : algorithms) {
                        testKnapsack(&ks, algo, saveToFile, saveToFile ? &log : nullptr);
                    }

                    if (saveToFile) log.close();
                    break;
                }

                case 4: { // Unbounded Knapsack
                    if (!saveToFile) {
                        std::cout << "Enter knapsack capacity: ";
                        std::cin >> capacity;
                        PrintBoxedText("Unbounded Knapsack Problem");
                    } else {
                        capacity = DEFAULT_CAPACITY_TESTED;
                        PrintBoxedText("Saving result to " + std::string(OUTPUT_UNBOUNDED_KNAPSACK));
                    }

                    knapsack::UnboundedKnapsack ks(capacity, items);
                    std::vector<std::string> algorithms = {"dp", "dp-1d", "recursive-memo"};

                    std::ofstream log;
                    if (saveToFile) {
                        std::string folder = "output";
                        if (!std::filesystem::exists(folder)) {
                            std::filesystem::create_directories(folder);
                        }   
                        std::string filename = folder + "/" + OUTPUT_UNBOUNDED_KNAPSACK;
                        log.open(filename);                
                        if (!log.is_open()) {
                            std::cerr << "Error: Failed to open output file: " << OUTPUT_UNBOUNDED_KNAPSACK << "\n";
                            break;
                        }
                        log << "Algorithm,TotalItems,ItemsValue,ItemsWeight,Quantity,Capacity,Time_ns,Result\n";
                    }

                    for (const auto& algo : algorithms) {
                        testKnapsack(&ks, algo, saveToFile, saveToFile ? &log : nullptr);
                    }

                    if (saveToFile) log.close();
                    break;
                }

                case 5: { // Multi-Dimensional Knapsack
                    int numDimensions;
                    if (!saveToFile) {
                        std::cout << "Enter number of dimensions: ";
                        std::cin >> numDimensions;
                        capacities.resize(numDimensions);
                        std::cout << "Enter capacities for each dimension: ";
                        for (int i = 0; i < numDimensions; ++i) {
                            std::cin >> capacities[i];
                        }
                        PrintBoxedText("Multi-Dimensional Knapsack Problem");
                    } else {
                        numDimensions = 1;
                        capacities.assign(numDimensions, DEFAULT_CAPACITY_TESTED);  // Default kapasitas tiap dimensi
                        PrintBoxedText("Saving result to " + std::string(OUTPUT_MULTI_DIMENSIONAL_KNAPSACK));
                    }

                    knapsack::MultiDimensionalKnapsack ks(capacities, items);
                    std::vector<std::string> algorithms = {"dp", "recursive-memo"};

                    if (!DISABLE_BRUTE_FORCE) {
                        algorithms.push_back("brute-force");                     
                    }

                    std::ofstream log;
                    if (saveToFile) {
                        std::string folder = "output";
                        if (!std::filesystem::exists(folder)) {
                            std::filesystem::create_directories(folder);
                        }   
                        std::string filename = folder + "/" + OUTPUT_MULTI_DIMENSIONAL_KNAPSACK;
                        log.open(filename);                                
                        if (!log.is_open()) {
                            std::cerr << "Error: Failed to open output file: " << OUTPUT_MULTI_DIMENSIONAL_KNAPSACK << "\n";
                            break;
                        }
                        log << "Algorithm,TotalItems,ItemsValue,ItemsWeight,Quantity,Capacity,Time_ns,Result\n";
                    }

                    for (const auto& algo : algorithms) {
                        testKnapsack(&ks, algo, saveToFile, saveToFile ? &log : nullptr);
                    }

                    if (saveToFile) log.close();
                    break;
                }

                case 6: { // Multi-Objective Knapsack
                    if (!saveToFile) {
                        std::cout << "Enter knapsack capacity: ";
                        std::cin >> capacity;
                        secondaryValues.resize(items.size());

                        if (inputChoice == 1) {
                            std::random_device rd;
                            std::mt19937 gen(rd());
                            std::uniform_int_distribution<> dist(1, 100);
                            for (size_t i = 0; i < items.size(); ++i) {
                                secondaryValues[i] = dist(gen);
                            }
                        } else {
                            std::cout << "Enter secondary values for each item:\n";
                            for (size_t i = 0; i < items.size(); ++i) {
                                std::cout << "Item " << i + 1 << ": ";
                                std::cin >> secondaryValues[i];
                            }
                        }
                        PrintBoxedText("Multi-Objective Knapsack Problem");
                    } else {
                        capacity = DEFAULT_CAPACITY_TESTED;
                        secondaryValues.resize(items.size(), 1); // default nilai
                        PrintBoxedText("Saving result to " + std::string(OUTPUT_MULTI_OBJECTIVE_KNAPSACK));
                    }

                    knapsack::MultiObjectiveKnapsack ks(capacity, items, secondaryValues);
                    std::vector<std::string> algorithms = {"dp", "lexicographic", "weighted-sum"};

                    if (!DISABLE_BRUTE_FORCE) {
                        algorithms.push_back("brute-force");                     
                    }

                    std::ofstream log;
                    if (saveToFile) {
                        std::string folder = "output";
                        if (!std::filesystem::exists(folder)) {
                            std::filesystem::create_directories(folder);
                        }   
                        std::string filename = folder + "/" + OUTPUT_MULTI_OBJECTIVE_KNAPSACK;
                        log.open(filename);                                                
                        if (!log.is_open()) {
                            std::cerr << "Error: Failed to open output file: " << OUTPUT_MULTI_OBJECTIVE_KNAPSACK << "\n";
                            break;
                        }
                        log << "Algorithm,TotalItems,ItemsValue,ItemsWeight,Quantity,Capacity,Time_ns,Result\n";
                    }

                    for (const auto& algo : algorithms) {
                        testKnapsack(&ks, algo, saveToFile, saveToFile ? &log : nullptr);
                    }

                    if (saveToFile) log.close();
                    break;
                }

                case 7: { // Multiple Knapsack
                    int numBags = 1;
                    std::vector<int> bagCapacities;

                    if (!saveToFile) {
                        std::cout << "Enter number of knapsacks: ";
                        std::cin >> numBags;

                        bagCapacities.resize(numBags);
                        std::cout << "Enter capacities for each knapsack: ";
                        for (int i = 0; i < numBags; ++i) {
                            std::cin >> bagCapacities[i];
                        }

                        PrintBoxedText("Multiple Knapsack Problem");
                    } else {
                        numBags = 1;
                        bagCapacities.assign(numBags, DEFAULT_CAPACITY_TESTED); // semua knapsack punya kapasitas default
                        PrintBoxedText("Saving result to " + std::string(OUTPUT_MULTIPLE_KNAPSACK));
                    }

                    knapsack::MultipleKnapsack ks(bagCapacities, items);

                    std::vector<std::string> algorithms = {"dp-each-bag", "greedy"};

                    std::ofstream log;
                    if (saveToFile) {
                        std::string folder = "output";
                        if (!std::filesystem::exists(folder)) {
                            std::filesystem::create_directories(folder);
                        }   
                        std::string filename = folder + "/" + OUTPUT_MULTIPLE_KNAPSACK;
                        log.open(filename);                                                                
                        if (!log.is_open()) {
                            std::cerr << "Error: Failed to open output file: " << OUTPUT_MULTIPLE_KNAPSACK << "\n";
                            break;
                        }
                        log << "Algorithm,TotalItems,ItemsValue,ItemsWeight,Quantity,Capacity,Time_ns,Result\n";
                    }

                    for (const auto& algo : algorithms) {
                        testKnapsack(&ks, algo, saveToFile, saveToFile ? &log : nullptr);
                    }

                    if (saveToFile) log.close();
                    break;
                }

                case 8: { // Quadratic Knapsack
                    if (!saveToFile) {
                        std::cout << "Enter knapsack capacity: ";
                        std::cin >> capacity;
                        PrintBoxedText("Quadratic Knapsack Problem");
                    } else {
                        capacity = DEFAULT_CAPACITY_TESTED;
                        PrintBoxedText("Saving result to " + std::string(OUTPUT_QUADRATIC_KNAPSACK));
                    }

                    knapsack::QuadraticKnapsack ks(capacity, items);

                    // Generate random interaction matrix Q
                    Q.resize(items.size(), std::vector<int>(items.size(), 0));
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_int_distribution<> dist(0, 10);

                    for (size_t i = 0; i < items.size(); ++i) {
                        for (size_t j = i + 1; j < items.size(); ++j) {
                            Q[i][j] = Q[j][i] = dist(gen); // symmetric
                        }
                    }
                    ks.setInteractionMatrix(Q);

                    std::vector<std::string> algorithms = {"greedy", "dp-approx"};

                    std::ofstream log;
                    if (saveToFile) {
                        std::string folder = "output";
                        if (!std::filesystem::exists(folder)) {
                            std::filesystem::create_directories(folder);
                        }   
                        std::string filename = folder + "/" + OUTPUT_QUADRATIC_KNAPSACK;
                        log.open(filename);                                                           
                        if (!log.is_open()) {
                            std::cerr << "Error: Failed to open output file: " << OUTPUT_QUADRATIC_KNAPSACK << "\n";
                            break;
                        }
                        log << "Algorithm,TotalItems,ItemsValue,ItemsWeight,Quantity,Capacity,Time_ns,Result\n";
                    }

                    for (const auto& algo : algorithms) {
                        testKnapsack(&ks, algo, saveToFile, saveToFile ? &log : nullptr);
                    }

                    if (saveToFile) log.close();
                    break;
                }

                case 9: { // Stochastic Knapsack 
                    if (!saveToFile) {
                        std::cout << "Enter knapsack capacity: ";
                        std::cin >> capacity;
                        PrintBoxedText("Stochastic Knapsack Problem");
                    } else {
                        capacity = DEFAULT_CAPACITY_TESTED;
                        PrintBoxedText(std::string("Saving result to ") + OUTPUT_STOCHASTIC_KNAPSACK);
                    }

                    knapsack::StochasticKnapsack ks(capacity, items);
                    
                    std::vector<std::vector<std::pair<int, double>>> weightProb;
                    weightProb.resize(items.size());

                    for (size_t i = 0; i < items.size(); ++i) {
                        int w1 = std::max(1, items[i].weight - 5);
                        int w2 = items[i].weight;
                        int w3 = items[i].weight + 5;

                        weightProb[i] = {
                            {w1, 0.3},
                            {w2, 0.4},
                            {w3, 0.3}
                        };

                        ks.setWeightProb(i, weightProb[i]);
                    }

                    std::vector<std::string> algorithms = {
                        "monte-carlo", "greedy-expected", "expected-dp"
                    };
                    
                    std::ofstream log;
                    if (saveToFile) {
                        std::string folder = "output";
                        if (!std::filesystem::exists(folder)) {
                            std::filesystem::create_directories(folder);
                        }   
                        std::string filename = folder + "/" + OUTPUT_STOCHASTIC_KNAPSACK;
                        log.open(filename);                                                           
                        if (!log.is_open()) {
                            std::cerr << "Error: Failed to open output file: " << OUTPUT_STOCHASTIC_KNAPSACK << "\n";
                            break;
                        }
                        log << "Algorithm,TotalItems,ItemsValue,ItemsWeight,Quantity,Capacity,Time_ns,Result\n";
                    }            

                    for (const auto& algo : algorithms) {
                        testKnapsack(&ks, algo, saveToFile, &log);
                    }

                    if (saveToFile) log.close();
                    break;
                }

                case 10: { // Multiple-Choice Knapsack
                    if (!saveToFile) {
                        std::cout << "Enter knapsack capacity: ";
                        std::cin >> capacity;

                        int numGroups;
                        std::cout << "Enter number of groups: ";
                        std::cin >> numGroups;
                        groups.resize(numGroups);

                        if (inputChoice == 1) {
                            std::random_device rd;
                            std::mt19937 gen(rd());
                            std::uniform_int_distribution<> groupDist(1, 5); // 1-5 items per group
                            std::uniform_int_distribution<> valueDist(1, 100);
                            std::uniform_int_distribution<> weightDist(1, 50);

                            for (int g = 0; g < numGroups; ++g) {
                                int groupSize = groupDist(gen);
                                for (int i = 0; i < groupSize; ++i) {
                                    groups[g].emplace_back(valueDist(gen), weightDist(gen));
                                }
                            }
                        } else {
                            for (int g = 0; g < numGroups; ++g) {
                                int groupSize;
                                std::cout << "Enter number of items in group " << g + 1 << ": ";
                                std::cin >> groupSize;
                                std::cout << "Enter items (value, weight) for group " << g + 1 << ":\n";
                                for (int i = 0; i < groupSize; ++i) {
                                    int v, w;
                                    std::cin >> v >> w;
                                    groups[g].emplace_back(v, w);
                                }
                            }
                        }

                        PrintBoxedText("Multiple-Choice Knapsack Problem");
                    } else {
                        // Default values for saveToFile mode
                        capacity = DEFAULT_CAPACITY_TESTED;
                        int numGroups = 5;
                        groups.resize(numGroups);

                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::uniform_int_distribution<> groupDist(1, 3); // 1-3 items per group
                        std::uniform_int_distribution<> valueDist(1, MAX_ITEMS_VALUE_TESTED);
                        std::uniform_int_distribution<> weightDist(1, MAX_ITEMS_WEIGHT_TESTED);

                        for (int g = 0; g < numGroups; ++g) {
                            int groupSize = groupDist(gen);
                            for (int i = 0; i < groupSize; ++i) {
                                groups[g].emplace_back(valueDist(gen), weightDist(gen));
                            }
                        }

                        PrintBoxedText(std::string("Saving result to ") + OUTPUT_MULTI_CHOICE_KNAPSACK);
                    }

                    knapsack::MultiChoiceKnapsack ks(capacity, groups);

                    std::vector<std::string> algorithms = {"dp", "greedy"};

                    if (!DISABLE_BRUTE_FORCE) {
                        algorithms.push_back("brute-force");                        
                    }

                    std::ofstream log(OUTPUT_MULTI_CHOICE_KNAPSACK);
                    if (!log.is_open()) {
                        std::cerr << "Failed to open file: " << OUTPUT_MULTI_CHOICE_KNAPSACK << "\n";
                        break;
                    }

                    log << "Algorithm,TotalItems,ItemsValue,ItemsWeight,Quantity,Capacity,Time_ns,Result\n";
                    for (const auto& algo : algorithms) {
                        testKnapsack(&ks, algo, saveToFile, &log);
                    }

                    if (saveToFile) log.close();
                    break;
                }

                case 11: { // Metaheuristic Approaches
                    if (!saveToFile) {
                        std::cout << "Enter knapsack capacity: ";
                        std::cin >> capacity;
                        PrintBoxedText("Metaheuristic Approaches to Knapsack Problem");
                    } else {
                        capacity = DEFAULT_CAPACITY_TESTED;
                        PrintBoxedText(std::string("Saving result to ") + OUTPUT_METAHEURISTIC_KNAPSACK);
                    }

                    knapsack::ZeroOneKnapsack ks(capacity, items); // Using 0/1 as base for metaheuristics
                    
                    std::vector<std::string> algorithms = {
                        "simulated-annealing", "ant-colony", "pso", "ilp", "constraint-programming"
                    };

                    std::ofstream log;
                    if (saveToFile) {
                        std::string folder = "output";
                        if (!std::filesystem::exists(folder)) {
                            std::filesystem::create_directories(folder);
                        }   
                        std::string filename = folder + "/" + OUTPUT_METAHEURISTIC_KNAPSACK;
                        log.open(filename);                                                           
                        if (!log.is_open()) {
                            std::cerr << "Error: Failed to open output file: " << OUTPUT_METAHEURISTIC_KNAPSACK << "\n";
                            break;
                        }
                        log << "Algorithm,TotalItems,ItemsValue,ItemsWeight,Quantity,Capacity,Time_ns,Result\n";
                    }  

                    for (const auto& algo : algorithms) {
                        testKnapsack(&ks, algo, saveToFile, &log, true);
                    }

                    if (saveToFile) log.close();
                    break;
                }

                default:
                    std::cout << "Invalid choice!\n";
                    break;  
            }
        } while (inputChoice == 4 && ++index <= 11);
    }

    std::string finishedMessage = R"(
    +=========================================+
    |                                         |
    |           PROGRAM FINISHED              |
    |                                         |
    +=========================================+


    )";

    int main() {
        signal(SIGINT, handleInterrupt);
        mainMenu();
        std::cout << finishedMessage;
        return 0;
    }
