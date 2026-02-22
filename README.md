# Matematisk modellerings opgaver
Repo til projekter til mat mod igennem semesteret

## UV

[UV](https://docs.astral.sh/uv/) er en hurtig Python package- og projektmanager skrevet i Rust. Det erstatter værktøjer som `pip`, `venv` og `pip-tools` med en enkelt, hurtig kommando.

### Installation

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Homebrew:**
```bash
brew install uv
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Brug i dette projekt

1. **Sync dependencies** (installerer alt fra `pyproject.toml` og opretter et virtuelt miljø automatisk):
   ```bash
   uv sync
   ```

2. **Kør et script med UV:**
   ```bash
   uv run python <script.py>
   ```

3. **Tilføj en ny dependency:**
   ```bash
   uv add <pakkenavn>
   ```

4. **Fjern en dependency:**
   ```bash
   uv remove <pakkenavn>
   ```

5. **Start Jupyter notebook:**
   ```bash
   uv run jupyter notebook
   ```