PROJECT   := ptest
CLIENTS   := 2
LLAMA_DIR := tiny-LlamaForCausalLM-3
SETUP_OUT := config.txt

.PHONY: all clean create run setup

all: run

setup:
	@python3 setup.py $(CLIENTS)

create:
	@cat $(SETUP_OUT) | xffl -dbg config -p $(PROJECT)

run:
	@xffl -dbg run -p $(PROJECT)

clean:
	@rm -rf $(PROJECT) $(LLAMA_DIR) $(SETUP_OUT)
	@docker ps -q | xargs -r docker stop