#!/bin/bash

#===============================================================================
# Ray Job Examples
# 
# This script runs Ray examples from the ray-examples/ directory:
# 1. Basic Ray task (hello world)
# 2. Parallel computation (map-reduce style)
# 3. Ray Actors (stateful workers)
# 4. Distributed data processing
# 5. Simple ML training example
#
# Usage:
#   ./ray-job-examples.sh [example]
#
# Examples:
#   ./ray-job-examples.sh hello        - Basic hello world
#   ./ray-job-examples.sh parallel     - Parallel tasks
#   ./ray-job-examples.sh actors       - Ray Actors demo
#   ./ray-job-examples.sh data         - Data processing
#   ./ray-job-examples.sh ml           - Simple ML training
#   ./ray-job-examples.sh all          - Run all examples
#   ./ray-job-examples.sh interactive  - Open shell in Ray pod
#===============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}/ray-examples"

# Source common library
source "${SCRIPT_DIR}/lib/common.sh"

#===============================================================================
# Example Functions
#===============================================================================

example_hello() {
    log_header "Example 1: Ray Hello World"
    
    cat << 'EOF'
This example shows:
- How to initialize Ray
- How to define a remote function with @ray.remote
- How to call remote functions and get results
- How tasks are distributed across the cluster

Source: ray-examples/01_hello.py
EOF
    
    log_info "Running hello world example..."
    run_python_file_in_ray "${EXAMPLES_DIR}/01_hello.py"
    log_success "Hello world example complete!"
}

example_parallel() {
    log_header "Example 2: Parallel Computation"
    
    cat << 'EOF'
This example shows:
- How Ray parallelizes work across the cluster
- Comparing sequential vs parallel execution time
- How tasks are distributed to workers

Source: ray-examples/02_parallel.py
EOF
    
    log_info "Running parallel computation example..."
    run_python_file_in_ray "${EXAMPLES_DIR}/02_parallel.py"
    log_success "Parallel computation example complete!"
}

example_actors() {
    log_header "Example 3: Ray Actors (Stateful Workers)"
    
    cat << 'EOF'
This example shows:
- How to create stateful workers with @ray.remote class
- Actors maintain state between method calls
- Actors can run on different nodes in the cluster

Source: ray-examples/03_actors.py
EOF
    
    log_info "Running actors example..."
    run_python_file_in_ray "${EXAMPLES_DIR}/03_actors.py"
    log_success "Actors example complete!"
}

example_data() {
    log_header "Example 4: Distributed Data Processing"
    
    cat << 'EOF'
This example shows:
- Processing data in parallel with Ray
- Map-reduce style operations
- Object store for sharing data between tasks

Source: ray-examples/04_data.py
EOF
    
    log_info "Running data processing example..."
    run_python_file_in_ray "${EXAMPLES_DIR}/04_data.py"
    log_success "Data processing example complete!"
}

example_ml() {
    log_header "Example 5: Distributed ML Training (Hyperparameter Search)"
    
    cat << 'EOF'
This example shows:
- Parallel hyperparameter search with Ray
- Training multiple models simultaneously
- Collecting and comparing results

Source: ray-examples/05_ml.py
EOF
    
    log_info "Running ML training example..."
    run_python_file_in_ray "${EXAMPLES_DIR}/05_ml.py"
    log_success "ML training example complete!"
}

#===============================================================================
# Interactive Shell
#===============================================================================

example_interactive() {
    log_header "Interactive Ray Shell"
    
    log_info "Opening interactive Python shell in Ray head pod..."
    log_info "Ray is already available. Try:"
    echo ""
    echo "  import ray"
    echo "  ray.init()"
    echo "  ray.cluster_resources()"
    echo ""
    log_info "Type 'exit()' to quit"
    echo ""
    
    local head_pod=$(get_ray_head_pod)
    kubectl exec -it "$head_pod" -- python3
}

#===============================================================================
# List Examples
#===============================================================================

list_examples() {
    log_header "Available Python Examples"
    
    echo "Location: ${EXAMPLES_DIR}/"
    echo ""
    
    if [ -d "$EXAMPLES_DIR" ]; then
        for f in "$EXAMPLES_DIR"/*.py; do
            if [ -f "$f" ]; then
                local filename=$(basename "$f")
                local desc=$(head -3 "$f" | grep -E '^"""' -A1 | tail -1 || echo "")
                echo "  - $filename"
                if [ -n "$desc" ]; then
                    echo "    $desc"
                fi
            fi
        done
    else
        echo "  (directory not found)"
    fi
    
    echo ""
    echo "Edit these files directly, then run with:"
    echo "  ./ray-job-examples.sh hello"
    echo "  ./ray-job-examples.sh parallel"
    echo "  etc."
}

#===============================================================================
# Run All Examples
#===============================================================================

run_all() {
    example_hello
    echo ""
    read -p "Press Enter to continue to next example..."
    
    example_parallel
    echo ""
    read -p "Press Enter to continue to next example..."
    
    example_actors
    echo ""
    read -p "Press Enter to continue to next example..."
    
    example_data
    echo ""
    read -p "Press Enter to continue to next example..."
    
    example_ml
    
    log_header "All Examples Complete!"
    
    cat << 'EOF'
You've seen:

1. Hello World     - Basic remote functions
2. Parallel        - Parallel task execution and speedup
3. Actors          - Stateful distributed workers
4. Data Processing - Map-reduce style operations
5. ML Training     - Parallel hyperparameter search

Next steps:
- Open Ray Dashboard: kubectl port-forward svc/ray-local-head-svc 8265:8265
- Try interactive mode: ./ray-job-examples.sh interactive
- Edit examples in: ray-examples/*.py
- Read Ray docs: https://docs.ray.io/

EOF
}

#===============================================================================
# Show Help
#===============================================================================

show_help() {
    cat << EOF
Ray Job Examples - Demonstrates Ray distributed computing concepts

Usage: ./ray-job-examples.sh [command]

Commands:
  hello        Run 01_hello.py - Basic hello world with @ray.remote
  parallel     Run 02_parallel.py - Parallel task execution and speedup
  actors       Run 03_actors.py - Ray Actors (stateful workers)
  data         Run 04_data.py - Distributed data processing (map-reduce)
  ml           Run 05_ml.py - Parallel ML hyperparameter search
  all          Run all examples sequentially
  interactive  Open Python shell in Ray pod
  list         List available Python example files

Examples:
  ./ray-job-examples.sh hello
  ./ray-job-examples.sh all
  ./ray-job-examples.sh interactive

Python files are in: ${EXAMPLES_DIR}/
Edit them directly and re-run to see changes!

Prerequisites:
  - Kind cluster with RayCluster running
  - Run ./setup-kind-with-registry.sh first

EOF
}

#===============================================================================
# Check Examples Directory
#===============================================================================

check_examples_dir() {
    if [ ! -d "$EXAMPLES_DIR" ]; then
        log_error "Examples directory not found: $EXAMPLES_DIR"
        exit 1
    fi
}

#===============================================================================
# Main
#===============================================================================

check_ray_cluster
check_examples_dir

case "${1:-help}" in
    hello)
        example_hello
        ;;
    parallel)
        example_parallel
        ;;
    actors)
        example_actors
        ;;
    data)
        example_data
        ;;
    ml)
        example_ml
        ;;
    all)
        run_all
        ;;
    interactive)
        example_interactive
        ;;
    list)
        list_examples
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
