#!/usr/bin/env bash
# Deploy theMnexus Lumen to AWS (Lambda arm64 + S3 + Function URL).
# Needs: AWS credentials (aws configure), Docker, AWS SAM CLI (brew install aws-sam-cli), uv.
# First run asks for stack name and region and saves them in samconfig.toml (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v aws >/dev/null || { echo "aws CLI missing: brew install awscli"; exit 1; }
command -v sam >/dev/null || { echo "SAM CLI missing: brew install aws-sam-cli"; exit 1; }
aws sts get-caller-identity --query Account --output text >/dev/null || { echo "no AWS credentials: aws configure"; exit 1; }

uv run pytest -q -p no:warnings
uv export --no-dev --no-emit-project --format requirements-txt -o infra/requirements.txt -q

sam build --template infra/template.yaml
if [ -f samconfig.toml ]; then
  sam deploy --resolve-image-repos
else
  sam deploy --guided --resolve-image-repos --capabilities CAPABILITY_IAM
fi

stack=$(grep -m1 'stack_name' samconfig.toml | cut -d'"' -f2)
url=$(aws cloudformation describe-stacks --stack-name "$stack" \
  --query "Stacks[0].Outputs[?OutputKey=='DemoUrl'].OutputValue" --output text)
echo "demo: $url"
curl -fsS "${url%/}/api/health" && echo
