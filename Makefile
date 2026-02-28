RUN=uv run

dev:
	git config blame.ignoreRevsFile .git-blame-ignore-revs
	$(RUN) prek install --install-hooks

lint:
	$(RUN) ruff check --fix --exclude third_party_models

format:
	$(RUN) ruff format --exclude third_party_models

.PHONY: dev lint format
