#!/bin/bash

#===============================================================================
# Common Library for Ray/Kubernetes Scripts
#
# This file contains shared functions used by multiple scripts.
# Source this file in your script:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"
#===============================================================================

#===============================================================================
# Colors
#===============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

#===============================================================================
# Logging Functions
#===============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

#===============================================================================
# Prerequisite Checks
#===============================================================================

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed. Please install it first."
        exit 1
    fi
}

check_docker_running() {
    if ! docker info &>/dev/null; then
        log_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
}

check_kubernetes_cluster() {
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot connect to Kubernetes cluster."
        echo "Make sure your cluster is running and kubectl is configured."
        exit 1
    fi
}

#===============================================================================
# Ray Cluster Functions
#===============================================================================

get_ray_head_pod() {
    kubectl get pod -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

check_ray_cluster() {
    local head_pod=$(get_ray_head_pod)
    if [ -z "$head_pod" ]; then
        log_error "No Ray head pod found. Is the RayCluster running?"
        echo "Run: kubectl get pods"
        exit 1
    fi
    log_info "Using Ray head pod: $head_pod"
}

run_python_in_ray() {
    local code="$1"
    local head_pod=$(get_ray_head_pod)
    kubectl exec -it "$head_pod" -- python3 -c "$code"
}

run_python_file_in_ray() {
    local file="$1"
    local head_pod=$(get_ray_head_pod)
    
    if [ ! -f "$file" ]; then
        log_error "Python file not found: $file"
        exit 1
    fi
    
    # Copy the file to the pod and run it
    local filename=$(basename "$file")
    kubectl cp "$file" "${head_pod}:/tmp/${filename}"
    kubectl exec -it "$head_pod" -- python3 "/tmp/${filename}"
}

#===============================================================================
# Pod Waiting Functions
#===============================================================================

wait_for_pods() {
    local namespace="${1:-default}"
    local timeout="${2:-300}"
    local label="${3:-}"
    
    log_info "Waiting for pods to be ready (timeout: ${timeout}s)..."
    
    local selector=""
    if [ -n "$label" ]; then
        selector="-l $label"
    fi
    
    local start_time=$(date +%s)
    local last_detail_time=0
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $timeout ]; then
            echo ""
            log_error "Timeout waiting for pods"
            kubectl get pods -n "$namespace" $selector
            return 1
        fi
        
        # Get current pod status
        local pod_info=$(kubectl get pods -n "$namespace" $selector --no-headers 2>/dev/null)
        local pod_count=$(echo "$pod_info" | grep -c . 2>/dev/null || echo "0")
        
        if [ "$pod_count" -eq 0 ]; then
            printf "\r  [%3ds] Waiting for pods to be created...                              " "$elapsed"
            sleep 2
            continue
        fi
        
        # Parse pod statuses
        local ready_count=0
        local total_count=0
        local status_summary=""
        local not_ready_pods=""
        
        while IFS= read -r line; do
            if [ -z "$line" ]; then continue; fi
            total_count=$((total_count + 1))
            
            local pod_name=$(echo "$line" | awk '{print $1}')
            local ready=$(echo "$line" | awk '{print $2}')
            local status=$(echo "$line" | awk '{print $3}')
            
            # Short pod name (last part)
            local short_name=$(echo "$pod_name" | rev | cut -d'-' -f1-2 | rev)
            
            if [ "$status" = "Running" ]; then
                local ready_containers=$(echo "$ready" | cut -d'/' -f1)
                local total_containers=$(echo "$ready" | cut -d'/' -f2)
                if [ "$ready_containers" = "$total_containers" ]; then
                    ready_count=$((ready_count + 1))
                    status_summary="${status_summary} ${short_name}:✓"
                else
                    status_summary="${status_summary} ${short_name}:${ready}"
                    not_ready_pods="${not_ready_pods} ${pod_name}"
                fi
            else
                status_summary="${status_summary} ${short_name}:${status}"
                not_ready_pods="${not_ready_pods} ${pod_name}"
            fi
        done <<< "$pod_info"
        
        # Print status update (overwrite previous line)
        printf "\r  [%3ds] Pods: %d/%d ready |%s                    " "$elapsed" "$ready_count" "$total_count" "$status_summary"
        
        # Every 15 seconds, show detailed events for not-ready pods
        if [ $((elapsed - last_detail_time)) -ge 15 ] && [ -n "$not_ready_pods" ]; then
            last_detail_time=$elapsed
            echo ""
            log_info "Current pod events:"
            for pod in $not_ready_pods; do
                local latest_event=$(kubectl get events -n "$namespace" --field-selector "involvedObject.name=$pod" --sort-by='.lastTimestamp' 2>/dev/null | tail -1)
                if [ -n "$latest_event" ]; then
                    echo "    $pod: $(echo "$latest_event" | awk '{print $NF}')"
                fi
            done
            # Also show if pulling images
            local pulling=$(kubectl describe pods -n "$namespace" $selector 2>/dev/null | grep -E "Pulling|Pulled" | tail -2)
            if [ -n "$pulling" ]; then
                echo "    Image status:"
                echo "$pulling" | while read -r line; do echo "      $line"; done
            fi
        fi
        
        # Check if all ready
        if [ "$ready_count" -eq "$total_count" ] && [ "$total_count" -gt 0 ]; then
            echo ""
            log_success "All pods are ready!"
            kubectl get pods -n "$namespace" $selector
            return 0
        fi
        
        sleep 3
    done
}

#===============================================================================
# Utility Functions
#===============================================================================

confirm_action() {
    local message="${1:-Continue?}"
    read -p "$message (y/n) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

get_script_dir() {
    cd "$(dirname "${BASH_SOURCE[1]}")" && pwd
}
