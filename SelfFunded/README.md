# Redis Cluster on Kubernetes

A self-contained, Helm-free Redis Cluster deployment with automatic
scaling. Keys are sharded across pods by hash slot, so any key always
routes to the pod that owns it — adding pods adds real capacity.

Three things are configurable:

| Setting | Variable | Default |
|---|---|---|
| Number of pods | `REDIS_REPLICAS` | `3` (minimum) |
| CPU per pod | `REDIS_CPU` | `500m` |
| Memory per pod | `REDIS_MEMORY` | `512Mi` |

## Quick start

```bash
./deploy.sh                                          # 3 pods, 500m CPU, 512Mi each
REDIS_REPLICAS=5 REDIS_CPU=1 REDIS_MEMORY=1Gi ./deploy.sh
```

`deploy.sh` renders the manifest template with `envsubst`, applies it,
waits for the pods, initializes the Redis Cluster (once — re-runs detect
an existing cluster and skip), and installs the auto-scaler.

## Scaling

```bash
REDIS_REPLICAS=5 ./scale.sh        # up or down
kubectl logs deploy/redis-scaler -f  # watch the reconcile happen
```

`scale.sh` only writes the desired count into the `redis-cluster-desired`
ConfigMap. The **redis-scaler** pod (a small controller, one per
namespace) reconciles the actual state to match every 15 s:

- **Scale up** — grows the StatefulSet, joins each new pod to the
  cluster (`redis-cli --cluster add-node`), then rebalances hash slots
  onto the empty nodes.
- **Scale down** — for each node being removed (highest ordinals
  first): drains its hash slots to the remaining nodes, removes it from
  the cluster, shrinks the StatefulSet, and deletes its PVC. No keys
  are lost.
- **Self-healing** — a scaler restart mid-scale-up is detected and
  finished on the next loop. Counts below 3 are ignored (Redis Cluster
  minimum).

> **Do not `kubectl scale statefulset redis` directly.** The ConfigMap
> is the source of truth and the scaler will reconcile the StatefulSet
> back to it. More importantly, a direct scale-down kills pods before
> their hash slots can be drained, losing the keys on them — `scale.sh`
> exists so draining always happens first.

## Changing CPU / memory

Re-run `deploy.sh` with new values:

```bash
REDIS_CPU=1 REDIS_MEMORY=1Gi ./deploy.sh
```

This performs a rolling restart. The cluster survives it: each node's
identity (`nodes.conf`) lives on its PVC, and `cluster-announce-ip`
re-announces the pod's new IP after every restart.

## Connecting

In-cluster clients bootstrap from `redis:6379` and **must be
cluster-aware** — they follow the cluster's slot map and talk directly
to pod IPs afterwards:

- redis-py: `RedisCluster(host="redis", port=6379)`
- ioredis: `new Redis.Cluster([{ host: "redis", port: 6379 }])`
- CLI: `redis-cli -c -h redis`

A plain single-node client will fail with `MOVED` redirect errors.

This setup is **in-cluster only**: the slot map hands out pod IPs,
which are not routable from outside Kubernetes. Exposing it externally
would require a per-pod NodePort/LoadBalancer with matching
`cluster-announce-ip` settings, or a cluster-aware proxy.

## Files

| File | Purpose |
|---|---|
| `redis-cluster.template.yaml` | ConfigMap (redis.conf), headless Service (per-pod DNS + cluster bus), client Service (`redis:6379`), StatefulSet with the `${REDIS_REPLICAS}`/`${REDIS_CPU}`/`${REDIS_MEMORY}` placeholders |
| `redis-scaler.yaml` | The auto-scaler: ServiceAccount + Role/RoleBinding, `redis-cluster-desired` ConfigMap (source of truth for pod count), reconcile script ConfigMap, and its Deployment |
| `deploy.sh` | Render + apply + one-time cluster init + scaler install |
| `scale.sh` | Set the desired pod count for the scaler to reconcile |

## Architecture notes

- **Every pod is a master** (`--cluster-replicas 0`): N pods = N shards
  of capacity. If a pod dies, its slice of the keyspace is unavailable
  until Kubernetes reschedules it; the data itself survives on the PVC
  (AOF persistence is on). If you need failover instead, each master
  needs a replica pod, doubling the pod count.
- **Storage** — each pod gets a 1Gi PVC (see `volumeClaimTemplates`).
  It primarily preserves cluster identity across restarts; size it up
  if you rely on AOF persistence for real data.
- **Requests = limits** — each pod gets exactly `REDIS_CPU` /
  `REDIS_MEMORY` (Guaranteed QoS), so capacity math is predictable.
- **Scaler permissions** — namespace-scoped Role only: pod exec,
  StatefulSet scale, ConfigMap read, PVC delete.

## Requirements

- `kubectl` pointed at your cluster, and `envsubst` (part of GNU
  gettext) on the machine running the scripts.
- A default StorageClass for the PVCs.
- Everything deploys into the current kubeconfig namespace.
