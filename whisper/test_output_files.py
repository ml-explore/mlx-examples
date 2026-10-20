import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_whisper import cli
from mlx_whisper.writers import get_writer


def transcript(text):
    return {"segments": [{"start": 0.0, "end": 1.0, "text": text}]}


class TestOutputFiles(unittest.TestCase):
    def test_dotted_names_with_each_writer(self):
        formats = ("txt", "vtt", "srt", "tsv", "json")
        for output_format in (*formats, "all"):
            with self.subTest(output_format=output_format):
                with tempfile.TemporaryDirectory() as output_dir:
                    writer = get_writer(output_format, output_dir)
                    writer(transcript("first recording"), "meeting.part1")
                    writer(transcript("second recording"), "meeting.part2")
                    extensions = formats if output_format == "all" else (output_format,)
                    expected = {
                        f"meeting.part{part}.{extension}"
                        for part in (1, 2)
                        for extension in extensions
                    }
                    self.assertEqual(
                        {path.name for path in Path(output_dir).iterdir()}, expected
                    )
                    for extension in extensions:
                        for part, text in (
                            (1, "first recording"),
                            (2, "second recording"),
                        ):
                            self.assertIn(
                                text,
                                (
                                    Path(output_dir) / f"meeting.part{part}.{extension}"
                                ).read_text(),
                            )

    def test_cli_output_names(self):
        cases = (
            (
                ("meeting.part1.wav", "meeting.part2.wav"),
                None,
                ("meeting.part1", "meeting.part2"),
            ),
            (("recording.wav",), "draft.v2", ("draft.v2",)),
            (("first.wav", "second.wav"), None, ("first", "second")),
        )
        for filenames, output_name, expected_names in cases:
            with self.subTest(filenames=filenames, output_name=output_name):
                with tempfile.TemporaryDirectory() as output_dir:
                    for filename in filenames:
                        argv = [
                            "mlx_whisper",
                            filename,
                            "--output-dir",
                            output_dir,
                            "--output-format",
                            "txt",
                            "--verbose",
                            "False",
                        ]
                        if output_name is not None:
                            argv.extend(("--output-name", output_name))
                        # Separate invocations isolate writer naming from the multi-input CLI loop.
                        with (
                            patch.object(sys, "argv", argv),
                            patch.object(
                                cli, "transcribe", return_value=transcript(filename)
                            ) as transcribe,
                        ):
                            cli.main()
                        self.assertEqual(transcribe.call_args.args[0], filename)
                    self.assertEqual(
                        {path.name for path in Path(output_dir).iterdir()},
                        {f"{name}.txt" for name in expected_names},
                    )
                    for name, filename in zip(expected_names, filenames):
                        self.assertEqual(
                            (Path(output_dir) / f"{name}.txt").read_text(),
                            filename + "\n",
                        )


if __name__ == "__main__":
    unittest.main()
