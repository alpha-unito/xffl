#!/bin/bash

cd $1

echo "--- Wall-clock time ---"
for i in *; do echo "$i: $(cat $i | grep "real" | awk '{ print $2; }' | tr s ' ' | awk -F m '{print $1 *60 + $2}' | sort -n | tail -1)"; done | sort

#echo "--- Training s/it ---"
#for i in *; do echo "$i: $(cat $i | grep "Training Epoch" | grep -o "\[[0-9]*:[0-9]*<[0-9]*:[0-9]*, *[[:digit:]]\.[[:digit:]][[:digit:]]s/it\]" | tr -s 's/it' ' ' | awk '{print $2}' | grep -o [[:digit:]]\.[[:digit:]][[:digit:]]  | sort -n | awk ' { a[i++]=$1; } END { print a[int(i/2)]; }')"; done | sort

#echo "--- Testing it/s ---"
#for i in *; do echo "$i: $(cat $i | grep "evaluating Epoch" | grep -o "\[[0-9]*:[0-9]*<[0-9]*:[0-9]*, *[[:digit:]]\.[[:digit:]][[:digit:]]it/s\]" | tr -s 's/it' ' ' | awk '{print $2}' | grep -o [[:digit:]]\.[[:digit:]][[:digit:]]  | sort -n | awk ' { a[i++]=$1; } END { print a[int(i/2)]; }')"; done | sort

echo "--- Training time time ---"
for i in *; do echo "$i: $(cat $i | grep "Training Epoch"  | grep -o "\[[0-9]*:[0-9]*<[0-9]*:[0-9]*, *[0-9]*.[0-9]*s/it\]" | tr -s '<' ' ' | tr -s '[' ' ' | awk '{print $1}'| awk -F: '{print $1 *60 + $2}' | sort -n | tail -1)"; done | sort
#for i in *; do echo "$i: $(cat $i | grep -o "Key: avg_epoch_time, Value: *[[:digit:]]*\.[[:digit:]]*" | awk '{print $4}')"; done | sort

echo "--- Testing time time ---"
for i in *; do echo "$i: $(cat $i | grep "evaluating Epoch" | grep -o "\[[0-9]*:[0-9]*<[0-9]*:[0-9]*, *[0-9]*.[0-9]*it/s\]" | tr -s '<' ' ' | tr -s '[' ' '| awk '{print $1}' | awk -F: '{print $1 *60 + $2}' | tail -1)"; done | sort

echo "--- Loss ---"
for i in *; do echo "$i: $(cat $i | grep "avg_eval_loss, Value:" | awk '{print $4}')"; done | sort

echo "--- Perplexity ---"
for i in *; do echo "$i: $(cat $i | grep "avg_eval_prep, Value:" | awk '{print $4}')"; done | sort

#paste <${WALL_CLOCK_TIME <$TRAINING_SIT <$TESTNG_SIT <$TRAINING_TIME <$TESTING_TIME <$LOSS <$PERPLEXITY | column -s $'\t' -t
