# 🌱🌍♻️ WP6.2 Multi-Level Heterogeneous ML Pipeline (GreenDIGIT Project)

*This work is funded from the European Union’s Horizon Europe research and innovation programme through the [GreenDIGIT project](https://greendigit-project.eu/), under the grant agreement No. [101131207](https://cordis.europa.eu/project/id/101131207)*.

<div style="display:flex;align-items:center;width:100%;margin-bottom:20px;">
  <img src="static/EN-Funded-by-the-EU-POS-2.png" alt="EU Logo" width="250px">
  <img src="static/cropped-GD_logo.png" alt="GreenDIGIT Logo" width="110px" style="margin-right:100px">
</div>


>**Disclaimer**: the information on this README is still temporary. The tools, architecture and other specifications are subject to change.

> Part of GreenDIGIT WP6.2 — Predictive AI for Federated Energy-Aware Workflows  
> Developed in collaboration with SoBigData RI, IFCA, DIRAC, and GreenDIGIT RIs and partners.


## Overview

This framework enables **real-time predictive modelling** across **Cloud, Grid and Network infrastructures** using **multi-level machine learning pipelines**. It ingests environmental and performance metrics (e.g. energy, CPU usage, workload profiles) from **distributed clusters and IoT devices**, processes them, and trains models to **forecast resource usage/availability and energy performance (CFP)**.

Deployed as part of the **GreenDIGIT WP6.2** research activities, this module integrates with:

- [WP6.1 Environmental Metric Publication System](#)
- [WP6.3 Energy-Aware Brokering Framework](#)
- UTH real-time IoT metrics infrastructure, data and workloads
- SoBigData RI metrics ecosystem
- IFCA and DIRAC records infrastructure

### To-dos (create tickets)
- [ ] DVC assets imported from remote storage (GDrive, AWS or server)
- ML model is quite simple. Things to improve.
  - [x] XGBoost, CatBoost (or other SoTA gradient boost tool-algo)
  - [x] Deep Learning: Convolutional Neural Network (LSTM, Temporal Convolution, Transformer) with PyTorch or TensorFlow
  - [ ] Use `scikit-learn-onnx` for more adaptability to edge-devices
  - [x] Integrate MQTT and/or Prometheus for edge-optimised messaging telemetry between devices (for the Edge)
- [ ] Metrics' ingestion: batch API from CNR
- [x] Metrics' ingestion: Kafka + MQTT + Flink/Spark + PostgreSQL/InfluxDB
- [ ] (Optional) Testbed implementation IoT with UTH
- M3L2 MVP inference-serving path.
  - [x] Typed broker-facing prediction schema
  - [x] Authenticated `/l2/predict` endpoint for L2 site-level forecast evidence
  - [x] Recurrent batch forecast refresh
  - [x] Idempotent training and workload-aware caching
  - [x] Basic HGBR baseline
  - [ ] Connect live EIMPS/MetricsDB execution-unit records ingestion
  - [ ] Add an operator workflow to register/configure a site before it is used for training
  - [ ] Add a DB-backed per-site training/forecast configuration, including whether each site is enabled, its characteristics, targets, minimum data requirements and any feature overrides
  - [ ] Evaluate model accuracy and compare HGBR, XGBoost, LSTM and ARIMA
  - [ ] Production load/latency testing
  - [ ] Finalise the contract with the WP6.3 Brokering service

## Models used

### Baseline: HistGradientBoostingRegressor (HGBR)
A tree-boosting model from `scikit-learn` used as the initial baseline. It operates on tabular, engineered features (lags, rolling statistics), is fast to train, handles missing values with an imputer, and gives a strong reference MAE/RMSE to beat.

### XGBoost (Gradient Boosted Trees)
High-performance gradient boosting on decision trees (histogram algorithm). Strong on heterogeneous tabular data, captures non-linear interactions well, robust to missing values, and typically competitive as a production baseline. In our pipeline it reads the same engineered features as the baseline and logs train/validation/test metrics plus validation curves to DVC.


### LSTM (sequence model)
A recurrent neural network that ingests sliding **time windows** shaped as `[samples, timesteps, features]`, therefore modelling temporal dependencies explicitly. Useful when recent history strongly determines near-future power. Requires normalised inputs and careful tuning; CPU training is slower than tree models.

### ARIMA/SARIMA
(Seasonal/) Autoregression Integrated Moving Average.
- [ ] TODO: write description.

> The pipeline includes a **champion selector** which chooses the best model by test error and writes `models/champion.json` for the inference service to load.

## Running the service and getting predictions

### 1) Train models and pick a winner
Ensure the DVC pipeline has produced models and metrics:
```bash
# From repo root
dvc repro          # runs ingest → featurise → train → validate_features → train_xgb → reshape_windows → train_lstm → select_champion
dvc metrics show   # compare baseline / xgb / lstm / (s)arima
```

### 2) Start the FastAPI service
```bash
uvicorn service.main:app --host 0.0.0.0 --port 8000
```

Health and model info:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/model
```

### 3) Request a prediction
For the M3L2 EUR-facing API, use the authenticated `/l2/predict` endpoint. Older model-specific prediction routes are compatibility-only and are hidden from the OpenAPI docs.
```bash
curl -X POST http://localhost:8000/l2/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
        "site_ids": ["SLICES-GR-UTH"],
        "horizon": "24h",
        "step": "1h",
        "workload": {"class": "batch", "cpu_hours": 8},
        "use_cache": true
      }'
```

#### Notes
- The service loads `models/champion.json`, so deployment remains model-agnostic.
- If your features use CIM paths or aliases, map them to the canonical na
mes server-side before padding missing inputs.
- For production, persist any scalers with the model and apply them inside the service, validate inputs, and secure the endpoint.

---

## Metrics Ingestion Services (Features)
- Ingest, Featurise and Train stages in-built as a pipeline (with DVC tracking).
- FastAPI server with hidden compatibility prediction routes and the M3L2 `/l2/predict` endpoint for EUR-facing forecasts.
- MQTT + Kafka + Flink streaming pipeline

## M3L2 MVP Production Path

The scoped MVP lives under `m3l2/`. It does four things:

- fetches execution records from CNR MetricsDB/EIMPS;
- stores normalised execution records plus Site Adapter profile/status snapshots in SQL;
- trains an `l2_site_status` model from training-compatible L2 Site Adapter status data every 6 hours;
- serves inferred availability, free resources, queue/provisioning delay, maintenance, and feasibility forecasts through FastAPI.
- exposes an EIMPS-style login page for 24-hour JWT tokens.

Run it:

```bash
cp .env.example .env
docker compose up --build -d
```

The Docker image uses `requirements-m3l2.txt`, a small runtime dependency set for the API. The broader `requirements.txt` still contains the heavier research stack.

Key M3L2 configuration:

- `M3L2_BATCH_LOOKBACK_HOURS`: rolling ingestion lookback window for scheduled MetricsDB pulls.
- `M3L2_TRAIN_INTERVAL_HOURS`: scheduled ingestion/training interval.
- `M3L2_FORECAST_REFRESH_MINUTES`: recurrent cached forecast refresh interval; default `15`.
- `M3L2_FORECAST_HORIZON_HOURS`: default horizon for recurrent forecast refresh.
- `M3L2_FORECAST_STEP_MINUTES`: default step for recurrent forecast refresh.
- `M3L2_MIN_TRAINING_RECORDS`: minimum usable site-telemetry records required before training.

Operational training and forecast settings can also be changed at runtime by a `site_admin` from:

```text
http://localhost:8000/ops/config/ui
```

The UI exposes non-secret service defaults and per-site overrides: automatic/manual training, training frequency,
training window, minimum usable telemetry records, forecast horizon/step/refresh cadence, selected model,
aggregation interval, submission cadence, staleness limits, and minimum coverage. Credentials and infrastructure
secrets remain server-side. The EIMPS checkbox is shown disabled as "Not yet available"; site registration and
telemetry submission do not require an EIMPS/MetricsDB execution-record mapping.

Use the CNR/EIMPS ingestion path:

```bash
# Service status and active model.
curl http://localhost:8000/health

# Fetch and store execution records.
curl -X POST http://localhost:8000/ingest/run \
  -H "Content-Type: application/json" \
  -d '{"start_ts":"2026-01-01T00:00:00Z","end_ts":"2026-01-02T00:00:00Z"}'

# Train manually with a site_admin token.
curl -X POST http://localhost:8000/ops/train \
  -H "Authorization: Bearer $TOKEN"

# Forecast L2 site-level evidence for the broker. Use a Bearer token from `/auth/token`.
curl -X POST http://localhost:8000/l2/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "broker-req-001",
    "candidate_site_ids": ["SLICES-GR-UTH", "OPENSTACK-DEMO"],
    "forecast_start_time": "2026-09-07T12:00:00Z",
    "horizon": "2h",
    "step": "1h",
    "workload": {
      "workload_id": "workload-123",
      "workload_type": "batch",
      "time_requirements": {
        "start_time": "2026-09-07T12:00:00Z",
        "duration": "1h",
        "deadline": "2026-09-07T16:00:00Z"
      },
      "resource_requirements": {
        "cpu": 4,
        "memory_gb": 8,
        "storage_gb": 20,
        "gpu": 0,
        "instances": 1,
        "flavour": "standard"
      },
      "metadata": {"project": "wp6-demo"},
      "extensions": {"application": "training-fixture"}
    },
    "cache": {"use_cache": true},
    "include_site_status": true
  }' \
  | jq .

# Inspect models and operational counters.
curl http://localhost:8000/models
curl http://localhost:8000/metrics
```

Set `M3L2_ENABLE_SCHEDULER=false` in `.env` to disable automatic ingestion and training.

Concise typed response shape:

```json
{
  "status": "ok",
  "request_id": "broker-req-001",
  "prediction_id": "generated-uuid",
  "generated_at": "2026-09-07T12:00:03Z",
  "valid_until": "2026-09-07T13:00:00Z",
  "model_name": "hist_gradient_boosting_mvp",
  "model_version": "l2-site-status-20260907T115900000000",
  "target": "l2_site_status",
  "forecast_start_time": "2026-09-07T12:00:00Z",
  "horizon": "2h",
  "step": "1h",
  "results": [
    {
      "site_id": "SLICES-GR-UTH",
      "target": "l2_site_status",
      "forecast": [
        {"ts": "2026-09-07T13:00:00Z", "value": 0.98, "unit": "ratio"}
      ],
      "energy_forecast": [],
      "site_status_forecast": [
        {
          "ts": "2026-09-07T13:00:00Z",
          "operational_status": "UP",
          "availability": 0.98,
          "free_cpu_capacity": 16,
          "queue_length": 1,
          "provisioning_delay_s": 30,
          "maintenance_flag": false,
          "inference_source": "model"
        }
      ],
      "capacity": {"compute_capacity": 32, "free_cpu_capacity": 16},
      "feasibility": {"status": "feasible", "reasons": []},
      "workload_estimates": {},
      "quality": {"forecast_quality": "baseline", "freshness": "cached", "confidence": "low"},
      "warnings": []
    }
  ],
  "warnings": []
}
```

If cached forecasts are absent or stale for the normalised workload signature, `/l2/predict` refreshes them with the active `l2_site_status` model and returns `cached_forecast_absent_refreshed` or `cached_forecast_stale_refreshed` in `warnings`. If no active model is registered, `/l2/predict` returns `503`; malformed typed workload/time/resource inputs return validation errors. Legacy prediction endpoints remain callable for compatibility but are hidden from the API docs.
Registered operator-facing site IDs such as `SLICES-GR-UTH` are resolved through `registered_sites.metadata.execution_records_site_id` before model inference. Responses include both `site_id` and `training_site_id`; unknown candidate sites return a clear `candidate_sites_not_found` response.

### L2 Site Adapter login and tokens

Open the login page:

```bash
http://localhost/auth/login
```

Token issuance is gated by `allowed_emails.txt` at the repository root. First login sets the password for an allowed email; subsequent logins must use the same password.
Protected L2 endpoints also check `SITE_ADAPTER_ALLOWED_EMAIL_DOMAINS`; include the domains of any non-institutional allowed emails, such as `gmail.com`, or set it empty to rely only on the explicit allow-list.

Supported allow-list formats:

```text
email@example.org
email@example.org,SITE-ID,reader
email@example.org,SITE-ID,publisher
email@example.org,SITE-ID,site_admin
email@example.org,SITE-ID,reader|publisher
```

JSON clients can request tokens directly:

```bash
curl -X POST http://localhost/auth/token \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@uth.gr","password":"change-me","site_id":"UTH-IOT","role":"site_admin"}'
```

Use the returned token on protected L2 endpoints:

```bash
curl http://localhost/auth/verify-token \
  -H "Authorization: Bearer <token>"
```

Tokens expire after 24 hours.

For local validation, `raw_data/summary_sites_15m.csv` is a 15-minute aggregate, not raw execution-unit data. Expected columns:

```text
bucket_15m,site_id,vo,activity,records,energy_wh,cfp_g,work,ncores
```

Load those aggregate rows into synthetic `execution_records`, derive training-compatible L2 site-status rows, and trigger training:

```bash
docker compose exec api python scripts/load_raw_aggregate_and_train.py
```

This stores the rows with `status="aggregated"` and `raw_json.source_file="raw_data/summary_sites_15m.csv"`. L2 Site Adapter snapshots submitted through `/l2/sites/{site_id}/snapshots` are also converted into training-compatible status rows automatically.

Forecast after training:

```bash
curl -X POST http://localhost:8000/l2/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"site_ids":null,"horizon":"24h","step":"1h","workload":{"class":"batch","cpu_hours":8},"use_cache":true}'
```

Format the response with `jq`:

```bash
curl -s -X POST http://localhost:8000/l2/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"site_ids":null,"horizon":"24h","step":"1h","workload":{"class":"batch","cpu_hours":8},"use_cache":true}' \
  | jq .
```

Brokering-oriented view:

```bash
curl -s -X POST http://localhost:8000/l2/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"site_ids":null,"horizon":"24h","step":"1h","workload":{"class":"batch","cpu_hours":8},"use_cache":true}' \
  | jq '.predictions[] | {
      site_id,
      inferred_availability: .forecast,
      inferred_status: .site_status_forecast,
      capacity,
      feasibility
    }'
```

Prediction responses include:

- `forecast`: inferred availability ratio over the requested horizon.
- `site_status_forecast`: model-inferred availability/free capacity, queue/provisioning delay, maintenance flag, and operational state.
- `capacity` and `feasibility`: broker-facing resource evidence derived from the inferred L2 forecast.

Mock Site Adapter availability data is available for local tests:

```text
raw_data/mock_site_profiles.json
raw_data/mock_site_status_snapshots.csv
```

Regenerate it from the aggregate validation CSV:

```bash
python3 scripts/generate_mock_site_availability.py --hours 168
```

Load it into the API database:

```bash
docker compose exec api python scripts/load_mock_site_status.py
```

Publish generic Site Adapter data directly:

```bash
curl -X POST "http://localhost:8000/site-profiles?adapter_type=iot" \
  -H "Content-Type: application/json" \
  -d '{"site_id":"SLICES-GR-UTH","ri_type":"network","location":"UTH","compute_capacity":32,"network_topology":"Mesh"}'

curl -X POST "http://localhost:8000/site-status?adapter_type=openstack" \
  -H "Content-Type: application/json" \
  -d '{"site_id":"OPENSTACK-DEMO","timestamp":"2026-01-01T00:00:00Z","total_vcpus":256,"free_vcpus":120,"total_gpus":8,"free_gpus":2,"pending_vms":4,"vm_provisioning_delay_s":180}'
```

`adapter_type` only selects supported input aliases (`generic`, `iot`, or `openstack`); it does not classify the
resource infrastructure. Use `ri_type` independently with one of `network`, `cloud`, or `grid` when the
classification is known. The submission endpoints validate the full batch before writing, return HTTP 422 with
field-specific errors for invalid values or conflicting aliases, and preserve unmapped fields in `extensions` with
submission warnings.

### Built-in mock L3 Site Adapter

The mock L3 adapter is built into the same FastAPI API container. You do not need a separate L3 container, process, broker, OpenStack monitor, or IoT monitor for the MVP validation flow.

What must be running:

- the `api` service from `docker compose up --build -d api`;
- Postgres from the same Compose stack;
- Nginx only if you want public HTTP on port `80`;
- an allowed email in `allowed_emails.txt` with `site_admin` for the site you want to register.

The mock L3 endpoints are public for inspection through Nginx:

```bash
curl http://localhost/mock-l3/sites/SLICES-GR-UTH/capabilities
curl http://localhost/mock-l3/sites/SLICES-GR-UTH/availability
curl "http://localhost/mock-l3/sites/SLICES-GR-UTH/usage?start=2026-06-30T00:00:00Z&end=2026-07-01T00:00:00Z&step=1h"
curl "http://localhost/mock-l3/sites/SLICES-GR-UTH/efficiency?start=2026-06-30T00:00:00Z&end=2026-07-01T00:00:00Z"
```

When registering a site, use the internal API URL as `adapter_base_url` because the L2 client runs inside the API container:

```text
http://127.0.0.1:8000/mock-l3/sites/SLICES-GR-UTH
```

Get a `site_admin` token:

```bash
TOKEN=$(curl -s -X POST http://localhost/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "email": "greendigit@uth.gr",
    "password": "your-password",
    "site_id": "SLICES-GR-UTH",
    "role": "site_admin"
  }' | jq -r '.access_token')
```

Register an L2 site against the built-in mock L3 adapter:

```bash
curl -X POST http://localhost/l2/sites/register \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "site_id": "SLICES-GR-UTH",
    "site_name": "SLICES-GR-UTH",
    "ri_type": "grid",
    "adapter_base_url": "http://127.0.0.1:8000/mock-l3/sites/SLICES-GR-UTH",
    "contact_email": "greendigit@uth.gr",
    "auth_type": "jwt",
    "metadata": {
      "eimps_site_name": "SLICES-GR-UTH",
      "execution_records_site_id": "site_e726c7cce5"
    }
  }'
```

For L2 Site Adapter registration, `ri_type` is one of `cloud`, `network`, or `grid`. EIMPS/MetricsDB connection is optional and not yet available in the operator UI, so registration no longer requires a matching execution-record site ID. If an execution-record mapping is known, it can still be kept in `metadata.execution_records_site_id` for prediction ID resolution. The current seeded local execution-record site IDs can be checked with:

```bash
docker compose exec postgres psql -U m3l2 -d m3l2 \
  -c "SELECT site_id, count(*) FROM execution_records GROUP BY site_id ORDER BY count(*) DESC LIMIT 20;"
```

Pull from the mock L3 adapter into L2DB:

```bash
curl -X POST http://localhost/l2/sites/SLICES-GR-UTH/pull \
  -H "Authorization: Bearer $TOKEN"
```

Read the stored snapshot:

```bash
curl http://localhost:8000/l2/sites/SLICES-GR-UTH/latest \
  -H "Authorization: Bearer $TOKEN"
curl http://localhost:8000/l2/sites/SLICES-GR-UTH/availability \
  -H "Authorization: Bearer $TOKEN"
```

Submitted site-level snapshots store the authenticated submitter email in `site_snapshots.submitted_by_email`.
Existing rows created before this field was added have `NULL` in that column.

Submitted telemetry is not automatically usable for training just because it was stored. The current training target is
`l2_site_status`; required inputs are `site_id`, `timestamp`, `operational_status`, `node_availability`,
`link_availability`, `free_cpu_capacity`, `queue_length`, and `load_index`. Rows are evaluated against the configured
aggregation interval, submission cadence, staleness limit, minimum usable record count, and minimum coverage. Unknown
fields remain in `extensions` and are excluded from training until explicitly mapped.

Check readiness and last-run state:

```bash
curl http://localhost:8000/ops/training/readiness \
  -H "Authorization: Bearer $TOKEN"
curl "http://localhost:8000/ops/training/readiness?site_id=SLICES-GR-UTH" \
  -H "Authorization: Bearer $TOKEN"
```

Check submitted snapshots in Postgres:

```bash
docker compose exec postgres psql -U m3l2 -d m3l2 \
  -c "SELECT id, site_id, ts, source, submitted_by_email FROM site_snapshots WHERE site_id = 'SLICES-GR-UTH' ORDER BY ts DESC LIMIT 20;"
```

Submit generated mock site metrics for one year from today at hourly cadence:

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "email": "greendigit@uth.gr",
    "password": "your-password",
    "site_id": "SLICES-GR-UTH",
    "role": "publisher"
  }' | jq -r '.access_token')

python3 scripts/submit_mock_site_snapshots.py \
  --base-url http://localhost:8000 \
  --site-id SLICES-GR-UTH \
  --days 365 \
  --step-minutes 60 \
  --dry-run

python3 scripts/submit_mock_site_snapshots.py \
  --base-url http://localhost:8000 \
  --site-id SLICES-GR-UTH \
  --days 365 \
  --step-minutes 60
```

Remove those example rows:

```bash
curl -X DELETE "http://localhost:8000/control/execution-records?source=raw_data/summary_sites_15m.csv&dry_run=false"
curl -X DELETE "http://localhost:8000/control/site-status?source=raw_data/summary_sites_15m.csv&dry_run=false"
```

If the container was built before this helper script existed, rebuild the API service:

```bash
docker compose up --build -d api
```

### MQTT + Kafka + Flink pipeline tutorial (development)
1. Install `docker-compose` with all containerised services (MQTT + Kafka).
```bash
cd streaming_service # you should see a docker-compose.yaml if you run ls -la
docker compose up -d --build
```

This will spin-up several services included in the compose file, including Kafka-UI, MQTT broker/subscriber and a Kafka bridge that ingests that service.
To see the logs from MQTT and Kafka respectively:
- `docker logs -f mqtt`
- `docker logs -f mqtt_to_kafka`

2. To start the synthetic workloads
```bash
# Go to the synthetic metrics' workload folder.
cd synthetic_metrics_service

# If you do not have the environment installed.
python -m venv .
source bin/activate

# Inside our environment:
python metrics_publisher.py
# From here you should see metrics being recurrently logged.
```

3. Kafka -> ELT (Flink SQL) -> Iceberg + MinIO
```sh
# Some useful command to list Kafka's topics, for debugging.
docker exec -it kafka kafka-topics.sh --bootstrap-server localhost:9092 --list

docker exec -it kafka /opt/bitnami/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
docker exec -it kafka /opt/bitnami/kafka/bin/kafka-configs.sh --bootstrap-server localhost:9092 \
  --entity-type topics --entity-name metrics.raw.stream --describe

```

4. Generating metrics (temporary)
```sh
# 1) Generate namespaces.json
python generate_namespaces.py --n 12

# 2a) Use existing namespace.json (no auto-generate nodes) (defaults SourceType=IoT)
python generate_synthetic_metrics.py --days 1 --freq-mins 3

# 2b) (Optional) autogenerate 12 IoT nodes
python generate_synthetic_metrics.py --autogen-nodes 12 --days 1 --freq-mins 3

# 3) Publish metrics
# 3-second fixed cadence, override payload timestamps to "now", IoT only
PACE_MODE=cadence CADENCE_S=3 OVERRIDE_TS=true SOURCE_TYPE=IoT \
BROKER=localhost PORT=1883 TOPIC_ROOT=greendigit QOS=1 \
python metrics_publisher.py

# Or: respect recorded Δts (scaled), keep original timestamps
PACE_MODE=replay_ts REPLAY_SPEED=2.0 OVERRIDE_TS=false SOURCE_TYPE=IoT \
python metrics_publisher.py

```

---

## Architecture
### Overview Architecture
![Overview Architecture](assets/gd_ecomep_overview_architecture.png)
- [ ] TODO: write description.

### Data Flow Architecture
![ECoMEP Data Flow](assets/gd_ecomep_pipeline.png)
- [ ] TODO: write description.

## Machine Learning Pipeline

### Ingestion & Preprocessing
- Collect metrics from edge nodes, sensors, and cluster logs
- Use **MQTT**, **Prometheus**, or **Kafka/NATS**
- Normalie, timestamp-align, and validate data

### Model Training
- Train using:
  - **Time Series Forecasting** (LSTM, Prophet)
  - **Regression/Classification** (XGBoost, RF)
  - **Energy/Latency Prediction**
- Tools: **PyTorch**, **TensorFlow**, **Scikit-learn**

### Real-Time Inference
- ONNX or TensorFlow Lite models served at edge
- Model registry: MLflow or DVC-based

---

## Folder Structure
- `.dvc/`, `dvc.yaml`, `dvc.lock` — DVC pipeline and metadata.
- `data/` — raw/clean/features/windows datasets managed by DVC.
- `metrics/` — JSON metrics tracked by DVC (baseline, XGBoost, LSTM).
- `models/` — trained artefacts (`baseline.joblib`, `xgb.joblib`, `lstm.pt`, `champion.json`).
- `scripts/` — pipeline scripts (`ingest.py`, `featurise.py`, `train.py`, `train_xgb.py`, `make_windows.py`, `train_lstm.py`, etc.).
- `service/` — FastAPI inference service (model-agnostic champion loader).
- `synthetic_metrics_service/`, `streaming_service/`, `ingest/` — data generation and streaming/ELT components.
- `assets/` — documentation assets.

<!-- ```bash
.
├── ingestion/             # Metric ingestion and connectors
├── preprocessing/         # Data cleaning and transformation
├── training/              # Training scripts and model tracking
├── inference/             # Model serving scripts (ONNX, Lite)
├── deployment/            # Helm charts, Dockerfiles
├── crate/                 # RO-Crate metadata, licences, schema
├── ro-crate-metadata.json
├── Dockerfile
├── requirements.txt
└── README.md
``` -->

## Outputs and Publications
Unified JSON or RO-Crate formatted metrics

- `/FETCH` endpoint compatible with WP6.1 publication system
- Optionally `POST`ed to:
    - cASO and Grid record services
    - CIM record registry with auth token

### Interoperability
- RO-Crate compliant for FAIR metadata
- Containerised for deployment in federated clusters
- Compatible with SoBigData metrics registry and Dirac grid APIs
- Modular, with pluggable ML models and data formats

## Citation
```
@software{GreenDIGIT_WP62,
  title = {Edge-Cloud Continuum Multi-Level Predictive Framework},
  author = {GreenDIGIT WP6.2 Contributors},
  year = {2025},
  version = {v1.0},
  url = {https://github.com/GreenDIGIT/WP6.2-Predictive-Framework}
}
```

## Contributors
Gonçalo Ferreira – UvA Researcher - WP6.2 Developer
- [ ] [Collaborators, Partners]

Supported by GreenDIGIT, SoBigData RI, IFCA, DIRAC, and CNR.

## Contact
For questions, integration requests or metric schema definitions, contact:

GreenDIGIT WP6.2 Team
📧 contact@greendigit.eu
🌐 greendigit.eu
