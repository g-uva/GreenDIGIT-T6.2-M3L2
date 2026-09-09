set -a; source .env; set +a;

API_BASE="${API_BASE:-https://gd3.lab.uvalight.net}"

REQUEST=$(curl -sS -X POST "$API_BASE/auth/token" \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg email "greendigit@uth.gr" \
    --arg password "$GD_EMAIL_PWD" \
    --arg site_id "SLICES-GR-UTH" \
    --arg role "publisher" \
    '{email:$email,password:$password,site_id:$site_id,role:$role}')")

TOKEN=$(echo "$REQUEST" | jq -r '.access_token')

python3 scripts/submit_mock_site_snapshots.py \
  --base-url "$API_BASE" \
  --site-id SLICES-GR-UTH \
  --days 365 \
  --step-minutes 60 \
  --token=$TOKEN

curl -sS "$API_BASE/health" | jq .

curl -sS -X POST "$API_BASE/l2/predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "manual-uth-public-001",
    "candidate_site_ids": ["SLICES-GR-UTH"],
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
