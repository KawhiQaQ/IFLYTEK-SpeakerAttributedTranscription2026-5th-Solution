.PHONY: check

check:
	git ls-files 'scripts/*.py' | xargs python -m py_compile
