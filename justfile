set positional-arguments

default:
    @just --list

# Install Python dependencies and start PostgreSQL.
setup: sync db-up

sync:
    uv sync --locked

sync-effnet:
    uv sync --locked --extra effnet

db-up:
    docker compose up -d --wait

db-status:
    docker compose ps

db-down:
    docker compose down

# Delete all stored analyses and restart PostgreSQL.
db-reset:
    docker compose down -v
    docker compose up -d --wait

check:
    uv run --locked ruff format --check .
    uv run --locked ruff check .
    uv run --locked ty check

test:
    uv run --locked python -m unittest -v test_main test_lab test_lab_database

verify: check test

test-db:
    RUN_DATABASE_TESTS=1 uv run --locked python -m unittest -v test_main test_lab_database

info:
    uv run --locked python -c 'import numba, torch; print("Numba:", numba.__version__); print("MPS:", torch.backends.mps.is_available())'

# Forward the FLAC path and any analyze options unchanged.
analyze +args:
    uv run --locked python main.py analyze "$@"

# Forward search options such as --track-id, --layer, and --limit.
search +args:
    uv run --locked python main.py search "$@"

# Print the names and shapes of exported arrays.
inspect-npz path:
    uv run --locked python -c 'import sys, numpy as np; a = np.load(sys.argv[1]); print({k: a[k].shape for k in a.files})' "$1"

# Preview or embed representative slices from a FLAC file or directory.
slices +args:
    uv run --locked python main.py slices "$@"

# Install the optional classifier runtime and classify existing slice IDs.
classify +args:
    uv run --locked --extra effnet python main.py classify "$@"

# Run the annotation app in a separate uv tool environment, bound to localhost.
# Pin follows upstream 1.23.0 source; native installation is still unverified here.
lab:
    mkdir -p qa/label-studio
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="{{justfile_directory()}}/qa" uv tool run --python 3.11 --from label-studio==1.23.0 label-studio start --internal-host 127.0.0.1 --port 8080 --data-dir "{{justfile_directory()}}/qa/label-studio"

lab-export +args:
    uv run --locked python lab.py export "$@"

lab-report +args:
    uv run --locked python lab.py report "$@"
