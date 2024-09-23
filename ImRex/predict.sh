#!/bin/bash

input_filepath='ImRex_test.csv'
predict_script_path='./src/scripts/predict/predict.py'
output_filepath='../result_path/ImRex_test.csv'
model_filepath='../Original_model/ImRex.h5'

python "$predict_script_path" --model "$model_filepath" --input "$input_filepath" --output "$output_filepath"

