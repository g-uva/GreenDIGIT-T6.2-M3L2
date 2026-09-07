set -a; source .env; set +a;

REQUEST=$(curl -sS -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d "$(jq -n \
    --arg email "greendigit@uth.gr" \
    --arg password "$GD_EMAIL_PWD" \
    --arg site_id "SLICES-GR-UTH" \
    --arg role "publisher" \
    '{email:$email,password:$password,site_id:$site_id,role:$role}')")

TOKEN=$(echo "$REQUEST" | jq -r '.access_token')

python3 scripts/submit_mock_site_snapshots.py \
  --base-url http://localhost:8000 \
  --site-id SLICES-GR-UTH \
  --days 365 \
  --step-minutes 60 \
  --token=$TOKEN