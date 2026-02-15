#!/usr/bin/env bash
#
# Unified API test script. Loads .env and runs all API tests.
#
# Usage:
#   ./test/run_all_tests.sh
#   conda activate science_star && ./test/run_all_tests.sh
#
# Requires: conda env "science_star" with project dependencies

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Use science_star conda env if available
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate science_star 2>/dev/null || true
fi

echo "🔍 Testing .env API keys..."
echo ""

PASS=0
FAIL=0

run_test() {
  local name="$1"
  local script="$2"
  echo "--- $name ---"
  if python "$script"; then
    ((PASS++)) || true
  else
    ((FAIL++)) || true
  fi
  echo ""
}

run_test "1. HF_TOKEN (Hugging Face)" "test/test_hf_token.py"
run_test "2. Search API (SERP/TAVILY)" "test/test_search_api.py"
run_test "3. Crawler API (Jina / crawl4ai)" "test/test_crawler_api.py"
run_test "4. LLM API (OpenAI-compatible)" "test/test_llm_api.py"

echo "================================"
echo "✅ Passed: $PASS"
echo "❌ Failed: $FAIL"
echo "================================"

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
exit 0
