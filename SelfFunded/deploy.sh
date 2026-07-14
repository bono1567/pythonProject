#!/usr/bin/env bash
# Deploy a Redis Cluster with configurable CPU, memory, and pod count.
# Keys are sharded across pods by hash slot, so any key routes to the
# pod that owns it (clients must be cluster-aware).
#
# Usage:
#   ./deploy.sh                                        # defaults: 3 pods, 500m CPU, 512Mi memory
#   REDIS_REPLICAS=5 REDIS_CPU=1 REDIS_MEMORY=1Gi ./deploy.sh
#
# Re-running with new REDIS_CPU/REDIS_MEMORY does a rolling restart; the
# cluster survives it (node identity lives in the PVC, IP changes are
# handled by cluster-announce-ip). To change pod count after the first
# deploy, use ./scale.sh instead — cluster slots must be rebalanced.
set -euo pipefail

export REDIS_REPLICAS="${REDIS_REPLICAS:-3}"
export REDIS_CPU="${REDIS_CPU:-500m}"
export REDIS_MEMORY="${REDIS_MEMORY:-512Mi}"

if [ "$REDIS_REPLICAS" -lt 3 ]; then
  echo "Redis Cluster needs at least 3 master nodes (got REDIS_REPLICAS=$REDIS_REPLICAS)" >&2
  exit 1
fi

cd "$(dirname "$0")"

echo "Deploying Redis Cluster: pods=${REDIS_REPLICAS} cpu=${REDIS_CPU} memory=${REDIS_MEMORY}"

envsubst '${REDIS_REPLICAS} ${REDIS_CPU} ${REDIS_MEMORY}' \
  < redis-cluster.template.yaml | kubectl apply -f -

echo "Waiting for pods to be ready..."
kubectl rollout status statefulset/redis --timeout=300s

# Initialize the cluster once: if redis-0 only knows itself, no cluster exists yet.
known_nodes=$(kubectl exec redis-0 -c redis -- redis-cli cluster info \
  | tr -d '\r' | awk -F: '/^cluster_known_nodes/{print $2}')
if [ "$known_nodes" -le 1 ]; then
  echo "Initializing cluster..."
  nodes=$(kubectl get pods -l app=redis \
    -o jsonpath='{range .items[*]}{.status.podIP}:6379 {end}')
  # shellcheck disable=SC2086
  kubectl exec redis-0 -c redis -- \
    redis-cli --cluster create $nodes --cluster-replicas 0 --cluster-yes
else
  echo "Cluster already initialized ($known_nodes known nodes)."
fi

kubectl exec redis-0 -c redis -- redis-cli cluster info | grep cluster_state

# Install the scaler (auto-reconciles cluster size to the
# redis-cluster-desired ConfigMap) and align its desired count.
kubectl apply -f redis-scaler.yaml
kubectl patch configmap redis-cluster-desired --type merge \
  -p "{\"data\":{\"replicas\":\"${REDIS_REPLICAS}\"}}"
