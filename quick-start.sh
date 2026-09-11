#!/bin/bash
# Secretary daemon quick start
# Usage: ./quick-start.sh

set -e

cd "$(dirname "$0")"

# 1. Copy example config if missing
if [ ! -f ~/.config/secretary/config.yaml ]; then
    echo "Creating config at ~/.config/secretary/config.yaml"
    mkdir -p ~/.config/secretary
    cp config.yaml ~/.config/secretary/config.yaml
fi

# 2. Copy .env example if missing
if [ ! -f .env ]; then
    echo "Creating .env from example"
    cp .env.example .env
    echo ">>> Edit .env to set OPENAI_API_KEY and other secrets"
fi

# 3. Deploy Qdrant (Mem0's vector store — no Mem0 server needed)
echo "Starting Qdrant..."
docker compose -f docker-compose.mem0.yaml up -d

# 4. Install systemd service
echo "Installing systemd service..."
cp secretary.service /etc/systemd/system/secretary.service
systemctl daemon-reload

echo ""
echo "=== Secretary daemon ready ==="
echo ""
echo "Next steps:"
echo "  1. Edit ~/.config/secretary/config.yaml (set opencode token if needed)"
echo "  2. Edit .env (set OPENAI_API_KEY for Mem0)"
echo "  3. systemctl start secretary"
echo "  4. systemctl enable secretary  (for boot)"
echo "  5. journalctl -u secretary -f   (to watch logs)"
echo ""
echo "Or run manually:"
echo "  .venv/bin/python main.py"
