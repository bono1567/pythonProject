#!/usr/bin/env bash
# Scale the Redis Cluster up OR down by setting the desired pod count.
# The redis-scaler pod picks up the change and does the rest:
#   up:   grow StatefulSet -> join new pods -> rebalance slots onto them
#   down: drain slots off removed nodes first -> then remove the pods
#
# Usage:
#   REDIS_REPLICAS=5 ./scale.sh
set -euo pipefail

if [ -z "${REDIS_REPLICAS:-}" ]; then
  echo "Usage: REDIS_REPLICAS=<n> ./scale.sh" >&2
  exit 1
fi
if [ "$REDIS_REPLICAS" -lt 3 ]; then
  echo "Redis Cluster needs at least 3 nodes (got REDIS_REPLICAS=$REDIS_REPLICAS)" >&2
  exit 1
fi

kubectl patch configmap redis-cluster-desired --type merge \
  -p "{\"data\":{\"replicas\":\"${REDIS_REPLICAS}\"}}"

echo "Desired replicas set to ${REDIS_REPLICAS}; the scaler will reconcile shortly."
echo "Watch progress with:  kubectl logs deploy/redis-scaler -f"