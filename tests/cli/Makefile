UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	SHASUM = sha256sum
endif
ifeq ($(UNAME_S),Darwin)
	SHASUM = shasum -a 256
endif

PROJECT=test 
WORKDIR="workdir"
MODEL_PATH="${WORKDIR}/llama3-1-8b"
TEMPDIR="${WORKDIR}/tmp"
RSA_KEY="${WORKDIR}/id_rsa"
ECDSA_CERT="${WORKDIR}/ecdsa"

checksum-config: setup generate-project test-files clean

clean:
	@rm -fr ${PROJECT} ${WORKDIR}

test-files:
	@test -f test/streamflow.yml || (echo "Error: test/streamflow.yml does not exist" && exit 1)
	@test -f test/cwl/config.yml || (echo "Error: test/cwl/config.yml does not exist" && exit 1)
	@find ${PROJECT} -not -name 'streamflow.yml' -not -name 'config.yml' -type f -print0 		| 		\
		sort -zd 					|		\
		xargs -r0 ${SHASUM} 		|		\
		${SHASUM} 					| 		\
		head -c 64

generate-project:
	@cat config-example.txt | xffl config -p ${PROJECT} --arguments --model-name llama3.1-8b --dataset-name clean_mc4_it

setup: clean
	@mkdir ${WORKDIR} ${TEMPDIR} ${MODEL_PATH}
	@echo "rsa" > ${RSA_KEY}
	@echo "cert" > ${ECDSA_CERT}