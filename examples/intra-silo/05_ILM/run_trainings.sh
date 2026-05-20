#! /bin/bash

LANGUAGES=( es eu it pl sl ) #nl de
max=${#LANGUAGES[@]}

export ILM_MODEL=BabyLM-130M


# MONOLINGUAL TRAININGS
export ILM_TYPE=mono
echo "TRAINING TYPE: $ILM_TYPE"

for TOKENIZER in bpe_8000 bpe_16000 bpe_32000 unigram_16000 bpe_trailing_ws_16000; do
	echo "	TOKENIZER: ${TOKENIZER}"
	for ((idxA=0; idxA<max; idxA++)); do
		echo "		Language: ${LANGUAGES[$idxA]}"
		OUT_FILE=/beegfs/home/gmittone/ILM/ILM/output/logs/slurm/${ILM_MODEL}_${ILM_TYPE}_${LANGUAGES[$idxA]}_${ILM_LANGUAGE_B}_${TOKENIZER}
		if [ -f ${OUT_FILE}.out ] || [ -f ${OUT_FILE}.err ]; then
			echo "Run logs already present - skipping execution"
		else
			echo "sbatch --export=MODEL_NAME=${ILM_MODEL},ILM_TYPE=${ILM_TYPE},ILM_LANGUAGE_A=${LANGUAGES[$idxA]},ILM_TOKENIZER=${TOKENIZER} --error ${OUT_FILE}.err --output ${OUT_FILE}.out --job-name ${ILM_MODEL}_${ILM_TYPE}_${LANGUAGES[$idxA]}_${ILM_LANGUAGE_B}_${TOKENIZER} ilm.slurm"
			#sbatch --export=MODEL_NAME=${ILM_MODEL},ILM_TYPE=${ILM_TYPE},ILM_LANGUAGE_A=${LANGUAGES[$idxA]},ILM_TOKENIZER=${TOKENIZER} --error ${OUT_FILE}.err --output ${OUT_FILE}.out --job-name ${ILM_MODEL}_${ILM_TYPE}_${LANGUAGES[$idxA]}_${ILM_LANGUAGE_B}_${TOKENIZER} ilm.slurm
		fi
	done
done
unset ILM_TYPE

# BILINGUAL TRAININGS
export ILM_TYPE=bilingual
echo "TRAINING TYPE: $ILM_TYPE"

for TOKENIZER in bpe_16000 bpe_32000 bpe_64000 unigram_32000 bpe_trailing_ws_32000; do
	echo "	TOKENIZER: ${TOKENIZER}"
	for ((idxA=0; idxA<max; idxA++)); do
		for ((idxB=idxA+1; idxB<max; idxB++)); do
			export ILM_LANGUAGE_A=${LANGUAGES[$idxA]}
			export ILM_LANGUAGE_B=${LANGUAGES[$idxB]}
			echo "		Languages: ${LANGUAGES[$idxA]} + ${LANGUAGES[$idxB]}"
			OUT_FILE=/beegfs/home/gmittone/ILM/ILM/output/logs/slurm/${ILM_MODEL}_${ILM_TYPE}_${LANGUAGES[$idxA]}_${LANGUAGES[$idxB]}_${TOKENIZER}
			if [ -f ${OUT_FILE}.out ] || [ -f ${OUT_FILE}.err ]; then
				echo "Run logs already present - skipping execution"
			else
				if [ ! -f /beegfs/home/gmittone/ILM/ILM/tokenizer/interlinguistic-language-modeling/tokenizer_ilm_${ILM_TYPE}_${ILM_LANGUAGE_A}_${ILM_LANGUAGE_B}_${TOKENIZER}/tokenizer.json ]; then
					echo "Missing tokenizer - skipping execution"
				else
					echo "sbatch --export=MODEL_NAME=${ILM_MODEL},ILM_TYPE=${ILM_TYPE},ILM_LANGUAGE_A=${LANGUAGES[$idxA]},ILM_LANGUAGE_B=${LANGUAGES[$idxB]},ILM_TOKENIZER=${TOKENIZER} --error ${OUT_FILE}.err --output ${OUT_FILE}.out --job-name ${ILM_MODEL}_${ILM_TYPE}_${LANGUAGES[$idxA]}_${LANGUAGES[$idxB]}_${TOKENIZER} ilm.slurm"
					#sbatch --export=MODEL_NAME=${ILM_MODEL},ILM_TYPE=${ILM_TYPE},ILM_LANGUAGE_A=${LANGUAGES[$idxA]},ILM_LANGUAGE_B=${LANGUAGES[$idxB]},ILM_TOKENIZER=${TOKENIZER} --error ${OUT_FILE}.err --output ${OUT_FILE}.out --job-name ${ILM_MODEL}_${ILM_TYPE}_${LANGUAGES[$idxA]}_${LANGUAGES[$idxB]}_${TOKENIZER} ilm.slurm
				fi
			fi
		done
	done
done
unset ILM_TYPE
