#!/usr/bin/env bash
# Mac M4 Max (64GB unified memory) - asynchronous oracle & DPO evaluator.
#
# Exposes Ollama on the LAN so the Linux orchestrator can offload System-2
# batch CoT generation, LLM-as-a-Judge validation, vision parsing and deep
# out-of-band escalations. Run once, then keep `ollama serve` running (the
# Ollama.app menu-bar app picks up the launchctl variables after a restart).
set -euo pipefail

MODEL="${S1_ORACLE_MODEL:-qwen3.8:27b}"
VISION_MODEL="${S1_ORACLE_VISION_MODEL:-}"

command -v ollama >/dev/null || { echo "Install Ollama first: https://ollama.com/download/mac"; exit 1; }

# Listen on all interfaces so the Linux host can reach port 11434.
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
# Keep the oracle resident; the factory calls it continuously.
launchctl setenv OLLAMA_KEEP_ALIVE "30m"
# Escalations + factory batches run concurrently against one loaded model.
launchctl setenv OLLAMA_NUM_PARALLEL "${OLLAMA_NUM_PARALLEL:-2}"
launchctl setenv OLLAMA_MAX_LOADED_MODELS "${OLLAMA_MAX_LOADED_MODELS:-2}"
launchctl setenv OLLAMA_FLASH_ATTENTION "1"
launchctl setenv OLLAMA_KV_CACHE_TYPE "q8_0"

echo "Pulling oracle model ${MODEL} ..."
ollama pull "${MODEL}"
if [[ -n "${VISION_MODEL}" ]]; then
  echo "Pulling vision model ${VISION_MODEL} ..."
  ollama pull "${VISION_MODEL}"
fi

echo
echo "Restart Ollama (quit the menu-bar app and reopen it, or run 'ollama serve')."
IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<mac-ip>")
echo "Then set on the Linux host (.env):"
echo "  S1_ORACLE_URL=http://${IP}:11434"
echo "  S1_ORACLE_MODEL=${MODEL}"
[[ -n "${VISION_MODEL}" ]] && echo "  S1_ORACLE_VISION_MODEL=${VISION_MODEL}"
echo "Verify from Linux:  curl http://${IP}:11434/api/tags"
