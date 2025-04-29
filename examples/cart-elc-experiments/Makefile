# run everything in series
SCRIPTS := $(wildcard *.py)

all: $(SCRIPTS)
	@for script in $^; do \
		echo "Running $$script..."; \
		python3 $$script || exit 1; \
	done
