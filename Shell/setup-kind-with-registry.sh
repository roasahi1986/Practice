#!/bin/bash

#===============================================================================
# Kind Cluster Setup with Local Registry
# 
# This script sets up:
# 1. A local Docker registry (persists across cluster recreations)
# 2. A Kind cluster configured to use the local registry
# 3. Pre-loaded images for KubeRay and Ray
# 4. KubeRay operator installation
# 5. A sample RayCluster deployment
#
# Usage:
#   ./setup-kind-with-registry.sh [command]
#
# Commands:
#   setup     - Full setup (default)
#   teardown  - Remove cluster and registry
#   status    - Check status of all components
#   images    - List images in local registry
#   load      - Load/push images to registry only
#===============================================================================

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common library
source "${SCRIPT_DIR}/lib/common.sh"

# Configuration
CLUSTER_NAME="ray-local"
REGISTRY_NAME="kind-registry"
REGISTRY_PORT="5001"
KIND_IMAGE="kindest/node:v1.26.0"

# Images to cache
KUBERAY_OPERATOR_IMAGE="quay.io/kuberay/operator:v1.2.2"
RAY_IMAGE="rayproject/ray:2.53.0"

#===============================================================================
# Registry Functions
#===============================================================================

create_registry() {
    log_info "Setting up local Docker registry..."
    
    # Check if registry already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${REGISTRY_NAME}$"; then
        if docker ps --format '{{.Names}}' | grep -q "^${REGISTRY_NAME}$"; then
            log_success "Registry '${REGISTRY_NAME}' is already running"
        else
            log_info "Starting existing registry '${REGISTRY_NAME}'..."
            docker start "${REGISTRY_NAME}"
            log_success "Registry started"
        fi
    else
        log_info "Creating new registry '${REGISTRY_NAME}'..."
        docker run -d \
            --restart=always \
            -p "127.0.0.1:${REGISTRY_PORT}:5000" \
            --name "${REGISTRY_NAME}" \
            registry:2
        log_success "Registry created and running on localhost:${REGISTRY_PORT}"
    fi
}

connect_registry_to_kind_network() {
    log_info "Connecting registry to Kind network..."
    
    # Check if kind network exists
    if ! docker network ls --format '{{.Name}}' | grep -q "^kind$"; then
        log_info "Kind network doesn't exist yet, will be created with cluster"
        return 0
    fi
    
    # Check if already connected
    if docker network inspect kind --format '{{range .Containers}}{{.Name}} {{end}}' 2>/dev/null | grep -q "${REGISTRY_NAME}"; then
        log_success "Registry already connected to Kind network"
    else
        docker network connect "kind" "${REGISTRY_NAME}" 2>/dev/null || true
        log_success "Registry connected to Kind network"
    fi
}

push_image_to_registry() {
    local source_image="$1"
    local registry_image="localhost:${REGISTRY_PORT}/$(echo $source_image | sed 's|.*/||')"
    
    log_info "Processing image: ${source_image}"
    
    # Check if image exists locally
    if ! docker image inspect "$source_image" &>/dev/null; then
        log_info "Pulling ${source_image}..."
        if ! docker pull "$source_image"; then
            log_error "Failed to pull ${source_image}"
            return 1
        fi
    else
        log_info "Image ${source_image} already exists locally"
    fi
    
    # Tag for local registry
    log_info "Tagging as ${registry_image}..."
    docker tag "$source_image" "$registry_image"
    
    # Push to local registry
    log_info "Pushing to local registry..."
    if docker push "$registry_image"; then
        log_success "Image pushed: ${registry_image}"
    else
        log_error "Failed to push ${registry_image}"
        return 1
    fi
}

load_images() {
    log_info "Loading images into local registry..."
    
    push_image_to_registry "$KUBERAY_OPERATOR_IMAGE"
    push_image_to_registry "$RAY_IMAGE"
    
    log_success "All images loaded into local registry"
}

list_registry_images() {
    log_info "Images in local registry (localhost:${REGISTRY_PORT}):"
    echo ""
    
    # List repositories
    local repos=$(curl -s "http://localhost:${REGISTRY_PORT}/v2/_catalog" 2>/dev/null | grep -o '"repositories":\[[^]]*\]' | sed 's/"repositories":\[//;s/\]//;s/"//g;s/,/ /g')
    
    if [ -z "$repos" ]; then
        log_warning "No images found in registry (or registry not running)"
        return
    fi
    
    for repo in $repos; do
        local tags=$(curl -s "http://localhost:${REGISTRY_PORT}/v2/${repo}/tags/list" 2>/dev/null | grep -o '"tags":\[[^]]*\]' | sed 's/"tags":\[//;s/\]//;s/"//g;s/,/ /g')
        for tag in $tags; do
            echo "  - localhost:${REGISTRY_PORT}/${repo}:${tag}"
        done
    done
    echo ""
}

#===============================================================================
# Kind Cluster Functions
#===============================================================================

create_kind_cluster() {
    log_info "Creating Kind cluster '${CLUSTER_NAME}'..."
    
    # Check if cluster already exists
    if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        log_warning "Cluster '${CLUSTER_NAME}' already exists"
        if confirm_action "Delete and recreate?"; then
            kind delete cluster --name "${CLUSTER_NAME}"
        else
            log_info "Using existing cluster"
            return 0
        fi
    fi
    
    # Create Kind config with registry
    local kind_config=$(mktemp)
    cat > "$kind_config" <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
containerdConfigPatches:
- |-
  [plugins."io.containerd.grpc.v1.cri".registry.mirrors."localhost:${REGISTRY_PORT}"]
    endpoint = ["http://${REGISTRY_NAME}:5000"]
nodes:
- role: control-plane
  extraPortMappings:
  - containerPort: 30000
    hostPort: 30000
    protocol: TCP
EOF
    
    # Create the cluster
    if kind create cluster --name "${CLUSTER_NAME}" --image "${KIND_IMAGE}" --config "$kind_config"; then
        log_success "Kind cluster created"
    else
        log_error "Failed to create Kind cluster"
        rm -f "$kind_config"
        return 1
    fi
    
    rm -f "$kind_config"
    
    # Connect registry to Kind network
    connect_registry_to_kind_network
    
    # Document the registry for the cluster
    log_info "Configuring registry for cluster..."
    kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: local-registry-hosting
  namespace: kube-public
data:
  localRegistryHosting.v1: |
    host: "localhost:${REGISTRY_PORT}"
    help: "https://kind.sigs.k8s.io/docs/user/local-registry/"
EOF
    
    log_success "Cluster configured to use local registry"
}

#===============================================================================
# KubeRay Installation
#===============================================================================

install_kuberay() {
    log_info "Installing KubeRay operator..."
    
    local charts_dir="${SCRIPT_DIR}/charts"
    local kuberay_chart="${charts_dir}/kuberay-operator.tgz"
    local kuberay_chart_alt="$HOME/Downloads/kuberay-operator.tgz"
    local chart_url="https://github.com/ray-project/kuberay-helm/releases/download/kuberay-operator-1.2.2/kuberay-operator-1.2.2.tgz"
    
    # Ensure charts directory exists
    mkdir -p "$charts_dir"
    
    # Check if already installed
    if helm list --short | grep -q "kuberay-operator"; then
        log_warning "KubeRay operator already installed"
        return 0
    fi
    
    # Find or download the chart
    local chart_to_use=""
    
    if [ -f "$kuberay_chart" ]; then
        log_info "Found chart in script directory: ${kuberay_chart}"
        chart_to_use="$kuberay_chart"
    elif [ -f "$kuberay_chart_alt" ]; then
        log_info "Found chart in Downloads: ${kuberay_chart_alt}"
        chart_to_use="$kuberay_chart_alt"
    else
        log_info "Chart not found locally, downloading..."
        if curl -4 -L --retry 3 --retry-delay 2 -o "$kuberay_chart" "$chart_url" 2>/dev/null; then
            log_success "Downloaded chart to ${kuberay_chart}"
            chart_to_use="$kuberay_chart"
        else
            log_warning "Direct download failed, trying Helm repo..."
            helm repo add kuberay https://ray-project.github.io/kuberay-helm/ 2>/dev/null || true
            helm repo update 2>/dev/null || true
            
            if helm install kuberay-operator kuberay/kuberay-operator --version 1.2.2 \
                --set image.repository="localhost:${REGISTRY_PORT}/operator" \
                --set image.tag="v1.2.2" 2>/dev/null; then
                log_success "Installed from Helm repo"
                log_info "Waiting for KubeRay operator to be ready..."
                wait_for_pods "default" 120 "app.kubernetes.io/name=kuberay-operator"
                log_success "KubeRay operator installed"
                return 0
            else
                log_error "Failed to install KubeRay. Please download manually:"
                echo "  curl -4 -L ${chart_url} -o ${kuberay_chart}"
                return 1
            fi
        fi
    fi
    
    # Install from local chart
    log_info "Installing from chart: ${chart_to_use}"
    if helm install kuberay-operator "$chart_to_use" \
        --set image.repository="localhost:${REGISTRY_PORT}/operator" \
        --set image.tag="v1.2.2"; then
        log_info "Waiting for KubeRay operator to be ready..."
        wait_for_pods "default" 120 "app.kubernetes.io/name=kuberay-operator"
        log_success "KubeRay operator installed"
    else
        log_error "Helm install failed"
        return 1
    fi
}

#===============================================================================
# RayCluster Deployment
#===============================================================================

create_raycluster() {
    log_info "Creating RayCluster..."
    
    local manifest_file="${SCRIPT_DIR}/manifests/raycluster.yaml"
    local registry_ray_image="localhost:${REGISTRY_PORT}/ray:2.53.0"
    
    # Check if manifest exists
    if [ ! -f "$manifest_file" ]; then
        log_error "RayCluster manifest not found: $manifest_file"
        return 1
    fi
    
    # Check if already exists
    if kubectl get raycluster ray-local &>/dev/null; then
        log_warning "RayCluster 'ray-local' already exists"
        if confirm_action "Delete and recreate?"; then
            kubectl delete raycluster ray-local
            sleep 5
        else
            return 0
        fi
    fi
    
    # Apply the manifest (substitute image if needed using envsubst or sed)
    log_info "Applying manifest: $manifest_file"
    sed "s|localhost:5001/ray:2.53.0|${registry_ray_image}|g" "$manifest_file" | kubectl apply -f -
    
    log_info "Waiting for RayCluster pods to be ready..."
    wait_for_pods "default" 180 "ray.io/cluster=ray-local"
    
    log_success "RayCluster created"
}

#===============================================================================
# Status and Info
#===============================================================================

show_status() {
    echo ""
    echo "==============================================================================="
    echo "                           CLUSTER STATUS"
    echo "==============================================================================="
    echo ""
    
    # Registry status
    log_info "Local Registry:"
    if docker ps --format '{{.Names}}' | grep -q "^${REGISTRY_NAME}$"; then
        echo "  Status: Running on localhost:${REGISTRY_PORT}"
        list_registry_images
    else
        echo "  Status: Not running"
    fi
    
    # Kind cluster status
    log_info "Kind Cluster:"
    if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        echo "  Status: Running"
        echo "  Context: kind-${CLUSTER_NAME}"
    else
        echo "  Status: Not running"
    fi
    
    # Kubernetes resources
    if kubectl cluster-info &>/dev/null; then
        echo ""
        log_info "Kubernetes Nodes:"
        kubectl get nodes -o wide
        echo ""
        log_info "Pods:"
        kubectl get pods -o wide
        echo ""
        log_info "RayClusters:"
        kubectl get rayclusters 2>/dev/null || echo "  No RayClusters found"
        echo ""
        log_info "Services:"
        kubectl get svc
    fi
    
    echo ""
    echo "==============================================================================="
    echo ""
}

show_next_steps() {
    echo ""
    echo "==============================================================================="
    echo "                              NEXT STEPS"
    echo "==============================================================================="
    echo ""
    echo "1. Access Ray Dashboard:"
    echo "   kubectl port-forward svc/ray-local-head-svc 8265:8265"
    echo "   Then open: http://localhost:8265"
    echo ""
    echo "2. Run Ray examples:"
    echo "   ./ray-job-examples.sh hello"
    echo "   ./ray-job-examples.sh all"
    echo ""
    echo "3. Check cluster status:"
    echo "   ./setup-kind-with-registry.sh status"
    echo ""
    echo "4. Teardown everything:"
    echo "   ./setup-kind-with-registry.sh teardown"
    echo ""
    echo "==============================================================================="
    echo ""
}

#===============================================================================
# Teardown
#===============================================================================

teardown() {
    log_warning "This will delete the Kind cluster and optionally the local registry"
    
    if ! confirm_action "Continue?"; then
        log_info "Aborted"
        return 0
    fi
    
    # Delete Kind cluster
    if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        log_info "Deleting Kind cluster..."
        kind delete cluster --name "${CLUSTER_NAME}"
        log_success "Cluster deleted"
    else
        log_info "Cluster '${CLUSTER_NAME}' not found"
    fi
    
    # Ask about registry
    if confirm_action "Also delete local registry? (Images will be lost)"; then
        if docker ps -a --format '{{.Names}}' | grep -q "^${REGISTRY_NAME}$"; then
            log_info "Deleting local registry..."
            docker rm -f "${REGISTRY_NAME}"
            log_success "Registry deleted"
        else
            log_info "Registry '${REGISTRY_NAME}' not found"
        fi
    else
        log_info "Registry preserved (images will be available for next cluster)"
    fi
    
    log_success "Teardown complete"
}

#===============================================================================
# Main
#===============================================================================

main() {
    echo ""
    echo "==============================================================================="
    echo "         Kind Cluster Setup with Local Registry"
    echo "==============================================================================="
    echo ""
    
    # Check prerequisites
    log_info "Checking prerequisites..."
    check_command docker
    check_command kind
    check_command kubectl
    check_command helm
    check_docker_running
    log_success "All prerequisites met"
    
    # Run setup steps
    create_registry
    load_images
    create_kind_cluster
    install_kuberay
    create_raycluster
    
    # Show status
    show_status
    show_next_steps
    
    log_success "Setup complete!"
}

# Parse command line arguments
case "${1:-setup}" in
    setup)
        main
        ;;
    teardown)
        teardown
        ;;
    status)
        show_status
        ;;
    images)
        list_registry_images
        ;;
    load)
        create_registry
        load_images
        ;;
    *)
        echo "Usage: $0 {setup|teardown|status|images|load}"
        echo ""
        echo "Commands:"
        echo "  setup     - Full setup (default)"
        echo "  teardown  - Remove cluster and registry"
        echo "  status    - Check status of all components"
        echo "  images    - List images in local registry"
        echo "  load      - Load/push images to registry only"
        exit 1
        ;;
esac
