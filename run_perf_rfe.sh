#!/bin/bash

NUM_FEATURES=7 # [1,12]
SCRIPT_NAME="script/4_rfe_xgboost.py" # Replace with your Python script name
OUTPUT_FILE="output/perf_output_4_rfe_xgboost_features_$NUM_FEATURES.txt" # The file where the perf output will be saved
COOL_DOWN_PERIOD=100 # Cool-down period in seconds
NUM_RUN=30

echo "Running perf for $SCRIPT_NAME $NUM_RUN times of $NUM_FEATURES features with a $COOL_DOWN_PERIOD second cool-down period..." >> $OUTPUT_FILE

# for i in {1..30}
for ((i = 1; i <= NUM_RUN; i++));
do

    echo "***Run #$i" >> $OUTPUT_FILE
    {
        perf stat -e power/energy-pkg/,power/energy-ram/ -a --repeat 1 python3 $SCRIPT_NAME $NUM_FEATURES &>> $OUTPUT_FILE
        if [ $? -ne 0 ]; then
            echo "Error: perf or Python script failed on run #$i" >> $OUTPUT_FILE
        fi
    } || {
        echo "Unexpected error occurred on run #$i" >> $OUTPUT_FILE
    }
    echo "" >> $OUTPUT_FILE
    echo "Cooling down for $COOL_DOWN_PERIOD seconds..."
    sleep $COOL_DOWN_PERIOD
    echo "Cool-down complete."
done

echo "Completed $NUM_RUN runs." >> $OUTPUT_FILE
