.PHONY: check dry-run

check:
	find scripts -maxdepth 1 -name '*.py' -print0 | xargs -0 python -m py_compile
	python scripts/verify_release.py .

dry-run:
	python scripts/train_models.py . --dry-run
	python scripts/run_inference.py . --dry-run --skip-preflight
