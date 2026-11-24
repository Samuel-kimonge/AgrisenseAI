# Deploying AgriSense AI (Docker, Compose, Kubernetes)

| File | Purpose |
|---|---|
| `Dockerfile` | CPU-only image, non-root (uid 10001), read-only-filesystem friendly. Targets: `runtime` (default) and `test` |
| `.dockerignore` | Keeps datasets, `.env`, caches and state out of the image |
| `constraints.txt` | Direct-dependency versions the test-suite ran against |
| `docker-compose.yml` | 2 replicas + nginx load balancer on one host |
| `deploy/nginx/nginx.conf` | Sticky (cookie) load balancer with websocket support |
| `deploy/k8s/base` | StatefulSet, Services, Ingress, HPA, PDB, NetworkPolicy, ConfigMap, namespace |
| `deploy/k8s/overlays/dev` | 1 replica, no autoscaler (kind / minikube / k3d) |
| `deploy/k8s/overlays/prod` | 3-8 replicas, TLS, 10 Gi volume per pod |

## 1. Docker

```bash
docker build -t agrisense-ai:0.1.0 .
docker run --rm -p 8501:8501 -v agrisense-data:/data agrisense-ai:0.1.0     # http://localhost:8501
docker build --target test -t agrisense-ai:test . && docker run --rm agrisense-ai:test   # run tests in the image
```

The image contains the code, the knowledge base and the trained model (`models/production`). It does **not**
contain the dataset, `.env` or any state. `/data` (SQLite history, saved photos, vector index) is the only writable path besides `/tmp`.
The vector index is rebuilt automatically on start if missing or if the knowledge base changed.

## 2. Docker Compose (two replicas + load balancer)

```bash
cp .env.example .env            # optional: LLM provider / API key
docker compose up --build -d    # http://localhost:8080
docker compose logs -f lb       # each line shows which replica served the request
docker compose down             # add -v to also delete the data volumes
```

## 3. Kubernetes

Prerequisites: a cluster, `kubectl`, the **ingress-nginx** controller, and **metrics-server** (for the autoscaler).

```bash
# Local cluster (kind) - one replica
docker build -t agrisense-ai:dev .
kind load docker-image agrisense-ai:dev
kubectl apply -k deploy/k8s/overlays/dev
kubectl -n agrisense port-forward svc/agrisense 8501:80        # http://localhost:8501

# Production
docker build -t registry.example.com/agrisense/agrisense-ai:0.1.0 . && docker push registry.example.com/agrisense/agrisense-ai:0.1.0
# edit deploy/k8s/overlays/prod/kustomization.yaml: image, domain (x2)
kubectl apply -k deploy/k8s/overlays/prod
kubectl -n agrisense get pods,svc,ingress,hpa

# Optional LLM key (never commit keys); set LLM_PROVIDER in the ConfigMap too
kubectl -n agrisense create secret generic agrisense-secrets --from-literal=ANTHROPIC_API_KEY='...'
kubectl -n agrisense rollout restart statefulset/agrisense
```

## How load balancing works (and why it is "sticky")

Streamlit keeps each browser session in memory inside one process, and photo uploads and the live websocket
belong to that process. A balancer that sends consecutive requests of one user to different replicas would
break uploads and the chat. So every layer pins a session to a replica:

* **Compose**: nginx hashes a random `asroute` cookie (new visitors are spread evenly; a replica failure re-hashes to a healthy one).
* **Kubernetes Ingress**: ingress-nginx cookie affinity, `persistent` mode (sessions are not moved when you scale out).
* **Kubernetes Service**: `ClientIP` affinity for direct in-cluster / port-forward access.

## State, scaling and known limits (please read)

* **History is per replica.** Each replica has its own SQLite database, photos and index on its own volume
  (SQLite must not run on a shared network filesystem, so no ReadWriteMany volume is used).
  With cookie stickiness a user keeps seeing the same history, but a user who loses the cookie (new browser, cleared cookies,
  or a replica removed by scale-down) may land on a replica with different history. For one shared history across all replicas the
  persistence layer has to move to a networked database such as PostgreSQL. That is a code change in `src/database` and is not done here.
* **Scaling out adds inference capacity, not shared state.** Scaling in ends the in-memory sessions on the removed pod; the
  autoscaler therefore scales down slowly (1 pod per 5 minutes after a 10-minute window). Volumes are retained
  (`persistentVolumeClaimRetentionPolicy: Retain`), so scaling back up reattaches the same history.
* **Capacity**: each pod requests 0.5 CPU / 1 GiB and may use 2 CPU / 2 GiB (model + Streamlit). `TORCH_THREADS` should not exceed the CPU limit.
* **Updating the model**: retrain, replace `models/production/`, rebuild and push a new image tag, then update the tag. The StatefulSet rolls pods one at a time.
* **Backups**: back up the PVCs (`data-agrisense-0`, ...) with your cluster's snapshot tooling. Photos and history are stored there.
* **The model and knowledge-base caveats in `README.md` still apply**: preliminary assessments only, no field validation, draft knowledge base and livestock rules.

## What was verified when this was written

Verified: `hadolint` (Dockerfile), `kustomize build` + `kubeconform --strict` on base/dev/prod (Kubernetes 1.30 schemas), `docker compose config`,
and the nginx config against two live Streamlit instances (even distribution, per-session stickiness, forged cookies ignored,
websocket upgrade, failover after killing a replica). The application was run as uid 10001 against a read-only copy of the code
tree (117 tests passed; model loaded; photo analysis, assistant and history worked; writes only to `/data`).

**Not verified** (no Docker daemon or cluster was available): an actual `docker build`, running the image, and applying the manifests to a live
cluster (scheduling, PVC provisioning, ingress-nginx cookie affinity, HPA behaviour). Expect to run the quick-start commands once and
adjust registry, domain, storage class and ingress class for your environment.
