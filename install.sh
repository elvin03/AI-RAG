#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OLLAMA_CONTAINER_NAME="${OLLAMA_CONTAINER_NAME:-ai-rag-ollama}"
OLLAMA_IMAGE="${OLLAMA_IMAGE:-docker.io/ollama/ollama:latest}"
OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_BASE_URL="http://127.0.0.1:${OLLAMA_PORT}"
OLLAMA_DATA_DIR="${OLLAMA_DATA_DIR:-${HOME}/.local/share/ollama}"
RAG_CHAT_MODEL="${RAG_CHAT_MODEL:-llama3.2:3b}"
RAG_EMBEDDING_MODEL="${RAG_EMBEDDING_MODEL:-nomic-embed-text}"
RAG_PORT="${RAG_PORT:-8765}"
RAG_DATA_DIR="${RAG_DATA_DIR:-${SCRIPT_DIR}/data}"
EDGE_PACK_REPO="${EDGE_PACK_REPO:-${RAG_DATA_DIR}/sources/edge-pack}"
NO_UPDATE=0
GUIDE_FILE=""
CONTAINER_ENGINE=""

usage() {
	cat <<'EOF'
Usage: ./install.sh [options]

Set up local Ollama models, index Edge Pack, run tests, and start the web app.
The Ollama service and RAG web app bind to loopback only.

Options:
  --repo PATH        Use an existing Edge Pack checkout or clone destination
  --port PORT        RAG web port (default: 8765)
  --ollama-port PORT Ollama host port (default: 11434)
  --no-update        Do not pull updates to an existing Edge Pack checkout
  --guide-file PATH  Index a saved getting-started HTML page
  --help             Show this help

Environment overrides:
  OLLAMA_IMAGE, OLLAMA_CONTAINER_NAME, OLLAMA_DATA_DIR
  RAG_CHAT_MODEL, RAG_EMBEDDING_MODEL, RAG_DATA_DIR
EOF
}

fail() {
	printf 'Error: %s\n' "$*" >&2
	exit 1
}

while (($#)); do
	case "$1" in
		--repo)
			(($# >= 2)) || fail "--repo requires a path"
			EDGE_PACK_REPO="$2"
			shift 2
			;;
		--port)
			(($# >= 2)) || fail "--port requires a number"
			RAG_PORT="$2"
			shift 2
			;;
		--ollama-port)
			(($# >= 2)) || fail "--ollama-port requires a number"
			OLLAMA_PORT="$2"
			OLLAMA_BASE_URL="http://127.0.0.1:${OLLAMA_PORT}"
			shift 2
			;;
		--no-update)
			NO_UPDATE=1
			shift
			;;
		--guide-file)
			(($# >= 2)) || fail "--guide-file requires a path"
			GUIDE_FILE="$2"
			shift 2
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			fail "unknown option: $1 (use --help)"
			;;
	esac
done

for port in "$RAG_PORT" "$OLLAMA_PORT"; do
	[[ "$port" =~ ^[0-9]{1,5}$ ]] || fail "ports must be numeric values from 1 to 65535"
	((10#$port >= 1 && 10#$port <= 65535)) || fail "ports must be numeric values from 1 to 65535"
done
[[ "$RAG_PORT" != "$OLLAMA_PORT" ]] || fail "RAG and Ollama must use different ports"
[[ "$RAG_CHAT_MODEL" =~ ^[A-Za-z0-9._:/-]+$ ]] || fail "invalid RAG_CHAT_MODEL"
[[ "$RAG_EMBEDDING_MODEL" =~ ^[A-Za-z0-9._:/-]+$ ]] || fail "invalid RAG_EMBEDDING_MODEL"

command -v python3 >/dev/null 2>&1 || fail "python3 is required"
command -v git >/dev/null 2>&1 || fail "git is required"
command -v curl >/dev/null 2>&1 || fail "curl is required"
python3 - <<'PY' || exit 1
import sqlite3
import sys

if sys.version_info < (3, 10):
    sys.exit("Error: Python 3.10 or newer is required")
try:
    sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE check_fts USING fts5(content)")
except sqlite3.OperationalError as error:
    sys.exit(f"Error: SQLite FTS5 is required: {error}")
PY

if ! curl --noproxy '*' --silent --show-error --fail --max-time 2 \
	"${OLLAMA_BASE_URL}/api/tags" >/dev/null 2>&1; then
	if command -v podman >/dev/null 2>&1; then
		CONTAINER_ENGINE="podman"
	elif command -v docker >/dev/null 2>&1; then
		CONTAINER_ENGINE="docker"
	else
		fail "Ollama is not available and neither Podman nor Docker is installed"
	fi

	if ! "$CONTAINER_ENGINE" image inspect "$OLLAMA_IMAGE" >/dev/null 2>&1; then
		printf 'Pulling Ollama runtime image %s...\n' "$OLLAMA_IMAGE"
		"$CONTAINER_ENGINE" pull "$OLLAMA_IMAGE"
	fi

	if "$CONTAINER_ENGINE" container inspect "$OLLAMA_CONTAINER_NAME" >/dev/null 2>&1; then
		container_running="$("$CONTAINER_ENGINE" inspect --format '{{.State.Running}}' "$OLLAMA_CONTAINER_NAME")"
		if [[ "$container_running" != "true" ]]; then
			printf 'Starting existing Ollama container %s...\n' "$OLLAMA_CONTAINER_NAME"
			"$CONTAINER_ENGINE" start "$OLLAMA_CONTAINER_NAME" >/dev/null
		fi
	else
		mkdir -p "$OLLAMA_DATA_DIR"
		if [[ "$CONTAINER_ENGINE" == "podman" ]]; then
			volume_spec="${OLLAMA_DATA_DIR}:/root/.ollama:Z"
		else
			volume_spec="${OLLAMA_DATA_DIR}:/root/.ollama"
		fi
		printf 'Starting loopback-only Ollama container %s...\n' "$OLLAMA_CONTAINER_NAME"
		"$CONTAINER_ENGINE" run --detach \
			--name "$OLLAMA_CONTAINER_NAME" \
			--restart unless-stopped \
			--publish "127.0.0.1:${OLLAMA_PORT}:11434" \
			--volume "$volume_spec" \
			"$OLLAMA_IMAGE"
	fi

	printf 'Waiting for Ollama at %s...\n' "$OLLAMA_BASE_URL"
	if ! curl --noproxy '*' --silent --show-error --fail \
		--retry 30 --retry-connrefused --retry-delay 1 --retry-max-time 60 --max-time 3 \
		"${OLLAMA_BASE_URL}/api/tags" >/dev/null; then
		fail "Ollama did not become ready; inspect with: $CONTAINER_ENGINE logs $OLLAMA_CONTAINER_NAME"
	fi
fi

pull_model() {
	local model="$1"
	local payload
	payload="$(printf '{"model":"%s","stream":false}' "$model")"
	printf 'Ensuring model %s is available locally...\n' "$model"
	curl --noproxy '*' --fail-with-body --silent --show-error \
		--request POST "${OLLAMA_BASE_URL}/api/pull" \
		--header 'Content-Type: application/json' \
		--data "$payload"
	printf '\n'
}

pull_model "$RAG_CHAT_MODEL"
pull_model "$RAG_EMBEDDING_MODEL"

export RAG_CHAT_BASE_URL="${OLLAMA_BASE_URL}/v1"
export RAG_CHAT_MODEL
export RAG_EMBEDDING_BASE_URL="${OLLAMA_BASE_URL}/v1"
export RAG_EMBEDDING_MODEL

if [[ -n "$GUIDE_FILE" ]]; then
	SYNC_ARGS=(sync --repo "$EDGE_PACK_REPO" --guide-file "$GUIDE_FILE")
else
	SYNC_ARGS=(sync --repo "$EDGE_PACK_REPO")
fi
if [[ "$NO_UPDATE" == 1 ]]; then
	SYNC_ARGS+=(--no-update)
fi
export RAG_DATA_DIR
python3 rag.py "${SYNC_ARGS[@]}"

python3 -m unittest discover -s tests

existing_status=""
if existing_status="$(curl --noproxy '*' --silent --show-error --fail --max-time 2 \
	"http://127.0.0.1:${RAG_PORT}/api/status" 2>/dev/null)"; then
	if printf '%s' "$existing_status" | python3 -c 'import json,sys; status=json.load(sys.stdin); sys.exit(0 if status.get("generation_configured") and status.get("embeddings_configured") else 1)'; then
		printf 'RAG app is already running with local models at http://127.0.0.1:%s\n' "$RAG_PORT"
		exit 0
	fi
	fail "an RAG app is already responding on port ${RAG_PORT}; it was left untouched. Choose another port with --port."
fi

if curl --noproxy '*' --silent --show-error --max-time 2 \
	"http://127.0.0.1:${RAG_PORT}/" >/dev/null 2>&1; then
	fail "port ${RAG_PORT} is occupied by another service; choose another port with --port"
fi

printf 'Starting Edge Pack Desk at http://127.0.0.1:%s (Ctrl-C stops the web app; Ollama stays running).\n' "$RAG_PORT"
exec python3 rag.py serve --port "$RAG_PORT"