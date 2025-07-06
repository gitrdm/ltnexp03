#!/bin/bash

# Kill process listening on a specific port
# Usage: ./killport.sh <port>
PORT=$1
if [ -z "$PORT" ]; then
  echo "Usage: $0 <port>"
  exit 1
fi

# Find and kill the process
PID=$(lsof -t -i:$PORT)
if [ -n "$PID" ]; then
  kill -9 $PID
  echo "Killed process $PID listening on port $PORT"
else
  echo "No process found listening on port $PORT"
fi
