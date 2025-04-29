SCRIPTS := $(wildcard *.py)

all: $(SCRIPTS)
	@for script in $^; do \
		echo "Running $$script..."; \
		python3 $$script & \
	done; \
	wait
