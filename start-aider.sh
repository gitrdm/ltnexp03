#!/bin/bash
# Optimized aider startup for ltnexp03 project

echo "🚀 Starting aider with optimized settings for large architecture..."
echo "📊 Token allocation: 32K repo map + 80K chat history + 131K context"
echo "🔧 Configuration: Protocols, contracts, neural-symbolic integration"
echo

# Start aider with optimal settings for your sophisticated codebase
aider \
  --map-tokens 32768 \
  --max-chat-history-tokens 81920 \
  --editor-model ollama/deepseek-coder-v2:latest \
  --cache-prompts \
  --auto-commits \
  --pretty \
  --show-diffs \
  "$@"
