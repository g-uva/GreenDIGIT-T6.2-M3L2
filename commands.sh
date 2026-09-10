set -a; source .env; set +a;

API_BASE="${API_BASE:-https://gd3.lab.uvalight.net}"
GONCALO_EMAIL="g.j.teixeiradepinhoferreira@uva.nl"
GONCALO_SITE="SARA-MATRIX"

# Getting the token
REQUEST=$(curl -sS -X POST "$API_BASE/auth/token" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg email "$GONCALO_EMAIL" \
    --arg password "$GONCALO_EMAIL_PWD" \
    --arg site_id "$GONCALO_SITE" \
    --arg role "site_admin" \
    '{email:$email,password:$password,site_id:$site_id,role:$role}')")

TOKEN=$(echo "$REQUEST" | jq -r '.access_token')

# Submitting synthetic mock site snapshots. NO NEED TO DO IT.
python3 scripts/submit_mock_site_snapshots.py \
  --base-url "$API_BASE" \
  --site-id $GONCALO_SITE \
  --days 365 \
  --step-minutes 60 \
  --token=$TOKEN

curl -sS "$API_BASE/health" | jq .

# Check the full service + per-site effective training/forecast configuration.
curl -sS "$API_BASE/ops/config?site_id=$GONCALO_SITE" \
  -H "Authorization: Bearer $TOKEN" | jq .

# Change one or two per-site settings. These apply to the next manual or scheduled training run.
curl -sS -X PATCH "$API_BASE/ops/config?site_id=$GONCALO_SITE" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "training_frequency_hours": 4,
    "min_usable_records": 24
  }' | jq .

# Check the whole config again after the override.
curl -sS "$API_BASE/ops/config?site_id=$GONCALO_SITE" \
  -H "Authorization: Bearer $TOKEN" | jq .

# Check whether the site has enough usable telemetry, then train a site-specific model.
curl -sS "$API_BASE/ops/training/readiness?site_id=$GONCALO_SITE" \
  -H "Authorization: Bearer $TOKEN" | jq .

curl -sS -X POST "$API_BASE/ops/train?site_id=$GONCALO_SITE" \
  -H "Authorization: Bearer $TOKEN" | jq .

curl -sS -X POST "$API_BASE/l2/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "manual-uth-public-001",
    "candidate_site_ids": ["SARA-MATRIX"],
    "horizon": "2h",
    "step": "1h",
    "workload": {
      "workload_id": "uth-test-1",
      "workload_type": "batch",
      "time_requirements": {"duration": "1h"},
      "resource_requirements": {"cpu": 4, "memory_gb": 8, "instances": 1}
    },
    "cache": {"use_cache": true},
    "include_site_status": true
  }' | jq '{
    status,
    target,
    model_name,
    model_version,
    cache_status: .cache.status,
    site: .results[0] | {
      site_id,
      training_site_id,
      registered_site_id,
      site_id_resolution,
      inferred_availability: .forecast,
      site_status_forecast,
      capacity,
      feasibility,
      warnings
    }
  }'

curl -sS -i -X POST "$API_BASE/l2/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "manual-missing-site-001",
    "candidate_site_ids": ["NO-SUCH-SITE"],
    "horizon": "1h",
    "step": "1h",
    "cache": {"use_cache": true}
  }'
