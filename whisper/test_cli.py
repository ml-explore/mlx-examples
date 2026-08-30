import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from mlx_whisper import cli


class TestCLI(unittest.TestCase):
    def test_multiple_audio_files_get_distinct_output_names(self):
        writer = Mock()
        transcribe = Mock(return_value={"segments": []})

        with tempfile.TemporaryDirectory() as output_dir:
            argv = [
                "mlx_whisper",
                "first.mp3",
                "second.mp3",
                "--output-dir",
                output_dir,
                "--verbose",
                "False",
            ]
            with (
                patch.object(cli, "get_writer", return_value=writer),
                patch.object(cli, "transcribe", transcribe),
                patch.object(sys, "argv", argv),
            ):
                cli.main()

        self.assertEqual(
            [call.args[1] for call in writer.call_args_list],
            ["first", "second"],
        )
        self.assertEqual(
            [call.args[0] for call in transcribe.call_args_list],
            ["first.mp3", "second.mp3"],
        )


if __name__ == "__main__":
    unittest.main()
