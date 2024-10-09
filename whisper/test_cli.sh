#!/bin/zsh -e

set -o err_exit

TEST_AUDIO="mlx_whisper/assets/ls_test.flac"
TEST_OUTPUT_DIR=$(mktemp -d -t mlx_whisper_cli_test)

# the control output - cli called with audio position arg
# expected output file name is ls_test.json
TEST_OUTPUT_NAME_FOR_ALL="--output-name arg is used for all output formats"
mlx_whisper "$TEST_AUDIO" \
    --output-dir "$TEST_OUTPUT_DIR" \
    --output-format all \
    --output-name '{basename}_transcribed' \
    --temperature 0 \
    --verbose=False
if /bin/ls ${TEST_OUTPUT_DIR}/ls_test_transcribed.{json,srt,tsv,txt,vtt} > /dev/null; then
    echo "[PASS] $TEST_OUTPUT_NAME_FOR_ALL"
else
    echo "[FAIL] $TEST_OUTPUT_NAME_FOR_ALL"
fi


TEST_OUTPUT_NAME_TEMPLATE="testing the output name template usage scenario"
for test_val in $(seq 10 10 60); do
    mlx_whisper "$TEST_AUDIO" \
        --output-name "{basename}_mwpl_${test_val}" \
        --output-dir "$TEST_OUTPUT_DIR" \
        --output-format srt \
        --max-words-per-line $test_val \
        --word-timestamps True \
        --verbose=False
    TEST_DESC="testing output name template while varying --max-words-per-line=${test_val}"
    if /bin/ls $TEST_OUTPUT_DIR/ls_test_mwpl_${test_val}.srt > /dev/null; then
        echo "[PASS] $TEST_DESC"
    else
        echo "[FAIL] $TEST_DESC"
    fi
done


TEST_STDIN_1="mlx_whisper produces identical output whether provided audio arg or stdin of same content"
/bin/cat "$TEST_AUDIO" | mlx_whisper - \
    --output-dir "$TEST_OUTPUT_DIR" \
    --output-format json \
    --temperature 0 \
    --verbose=False
if diff "${TEST_OUTPUT_DIR}/content.json" "${TEST_OUTPUT_DIR}/ls_test_transcribed.json"; then
    echo "[PASS] $TEST_STDIN_1"
else
    echo "[FAIL] $TEST_STDIN_1"
    echo "Check unexpected output in ${TEST_OUTPUT_DIR}"
fi

TEST_STDIN_2="mlx_whisper produces identical output when stdin comes via: cmd < file"、
mlx_whisper - \
    --output-name '{basename}_transcribed' \
    --output-dir "$TEST_OUTPUT_DIR" \
    --output-format tsv \
    --temperature 0 \
    --verbose=False < "$TEST_AUDIO"
if diff "${TEST_OUTPUT_DIR}/content_transcribed.tsv" "${TEST_OUTPUT_DIR}/ls_test_transcribed.tsv"; then
    echo "[PASS] $TEST_STDIN_2"
else
    echo "[FAIL] $TEST_STDIN_2"
    echo "Check unexpected output in ${TEST_OUTPUT_DIR}"
fi

echo "Outputs can be verified in ${TEST_OUTPUT_DIR}"
