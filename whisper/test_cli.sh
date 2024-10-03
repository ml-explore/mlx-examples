#!/bin/zsh -e

set -o err_exit

TEST_AUDIO="mlx_whisper/assets/ls_test.flac"

# when not receiving stdin, check audio arg is required
TEST_1="mlx_whisper requires audio position arg when not provided with stdin"
if mlx_whisper 2>&1 | grep "the following arguments are required: audio" > /dev/null; then
    echo "[PASS] $TEST_1"
else
    echo "[FAIL] $TEST_1"
fi

TEST_2="mlx_whisper does not require audio position arg when provided with stdin"
if ! (/bin/cat "$TEST_AUDIO" | mlx_whisper --help | /usr/bin/grep "Audio file(s) to transcribe") > /dev/null; then
    echo "[PASS] $TEST_2"
else
    echo "[FAIL] $TEST_2"
fi

TEST_3="mlx_whisper accepts optional --input-name arg"
if (mlx_whisper --help | /usr/bin/grep "\-\-input-name") > /dev/null; then
    echo "[PASS] $TEST_3"
else
    echo "[FAIL] $TEST_3"
fi

TEST_OUTPUT_DIR=$(mktemp -d -t mlx_whisper_cli_test)

# the control output - cli called with audio position arg
# expected output file name is ls_test.json
mlx_whisper "$TEST_AUDIO" --output-dir "$TEST_OUTPUT_DIR" --output-format all --temperature 0 --verbose=False


TEST_STDIN_1="mlx_whisper produces identical output whether provided audio arg or stdin of same content"
# method stdin - output file is content.json (default --input-name is content when not provided)
/bin/cat "$TEST_AUDIO" | mlx_whisper --output-dir "$TEST_OUTPUT_DIR" --output-format json --temperature 0 --verbose=False
if diff "${TEST_OUTPUT_DIR}/content.json" "${TEST_OUTPUT_DIR}/ls_test.json"; then
    echo "[PASS] $TEST_STDIN_1"
else
    echo "[FAIL] $TEST_STDIN_1"
    echo "Check unexpected output in ${TEST_OUTPUT_DIR}"
fi

TEST_STDIN_2="mlx_whisper produces identical output when stdin comes via: cmd < file"、
mlx_whisper --input-name stdin_test_2 --output-dir "$TEST_OUTPUT_DIR" --output-format tsv --temperature 0 --verbose=False < "$TEST_AUDIO"
if diff "${TEST_OUTPUT_DIR}/stdin_test_2.tsv" "${TEST_OUTPUT_DIR}/ls_test.tsv"; then
    echo "[PASS] $TEST_STDIN_2"
else
    echo "[FAIL] $TEST_STDIN_2"
    echo "Check unexpected output in ${TEST_OUTPUT_DIR}"
fi
