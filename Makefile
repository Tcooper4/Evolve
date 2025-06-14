.PHONY: test test-fix clean install\n\n# Python interpreter\nPYTHON = python\n\n# Test commands\ntest:\n\t run_tests.py\n\ntest-fix:\n\t run_tests.py --fix\n\n# Clean up Python cache files and coverage reports\nclean:\n\tfind . -type d -name \
__pycache__\ -exec rm -r {} +\n\tfind . -type d -name \*.egg-info\ -exec rm -r {} +\n\tfind . -type d -name \.pytest_cache\ -exec rm -r {} +\n\tfind . -type d -name \htmlcov\ -exec rm -r {} +\n\tfind . -type f -name \*.pyc\ -delete\n\tfind . -type f -name \.coverage\ -delete\n\n# Install development dependencies\ninstall:\n\tpip install -r requirements.txt\n\tpip install -r requirements-dev.txt\n\n# Help command\nhelp:\n\t@echo \Available
commands:\\n\t@echo \
make
test
-
Run
tests
with
coverage\\n\t@echo \
make
test-fix
-
Run
tests
and
fix
code
formatting\\n\t@echo \
make
clean
-
Clean
up
cache
files
and
coverage
reports\\n\t@echo \
make
install
-
Install
development
dependencies\
