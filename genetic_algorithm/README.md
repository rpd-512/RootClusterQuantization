# Fitness evaluator for underdetermined systems F(x)=0

Header-only C++17. Dependency: Eigen 3.3+ (header-only).

## Build
    sudo apt install libeigen3-dev cmake        # or: brew install eigen cmake
    cmake -S . -B build && cmake --build build && ./build/genetic_algorithm
    # no CMake:
    g++ -std=c++17 -O2 -I. -I/usr/include/eigen3 main.cpp includes/example_generator.cpp -o genetic_algorithm

## Layout
    CMakeLists.txt  main.cpp  README.md
    includes/  types.hpp  equation_system.hpp  metrics.hpp  discrete_solver.hpp
               continuous_solver.hpp  evaluator.hpp  example_generator.cpp (stub generator for the demo)

## Use
    #include "includes/evaluator.hpp"
    ProblemSpec spec; spec.n = 6; spec.num_equations = 4;          // n and n-m
    spec.domain = Domain::integer_range(6, 0, 3);                  // or discrete_per_variable / continuous_box
    spec.original = Point{...};                                    // optional
    FitnessEvaluator eval(spec, Config{});
    FitnessResult r = eval(system);                                // system: EquationSystem / CallableSystem
    double f = r.fitness;                                          // average pairwise distance

## Semantics
| Case | completeness | root_count | metrics |
|---|---|---|---|
| Discrete, finished | Exhaustive | exact | Exact if all roots stored (<= max_stored_roots), else Estimate (uniform reservoir) |
| Discrete, budget hit | Truncated | lower bound | Estimate (roots found only) |
| Continuous | Sampled | distinct samples (not cardinality) | Estimate |
| < 2 roots | — | — | pairwise metrics Undefined; fitness = Config::undefined_fitness, fitness_defined = false |

## Limitations
- Discrete cost is O(prod |alphabet|) worst case; declare equation supports to prune.
- Continuous: LM multistart from uniform box samples; samples are not uniform on the solution set, can miss
  disconnected components or roots on box boundaries, and min pairwise is floored near `distinct_tol`.
  No result is a proof of infeasibility. Functions must be finite on the box (NaN/inf residuals count as failures).
- Pairwise statistics are O(k^2 n) over at most `max_stored_roots` roots.
