
checksum-config:
	cat config-example.txt |  xffl config -p test --arguments --model-name llama3.1-8b --dataset-name clean_mc4_it
	find test -type f \( -exec sha256sum "${PWD}"/{} \; \) | awk '{print $1}' | sort | sha256sum | head -c 64