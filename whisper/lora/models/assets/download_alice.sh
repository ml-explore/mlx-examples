#!/bin/bash

audio_file=$HOME/.cache/whisper/alice.mp3
echo $audio_file
zipf=alice_in_wonderland_librivox_64kb_mp3.zip
url=https://www.archive.org/download/alice_in_wonderland_librivox/
curl -LO $url/$zipf
unzip $zipf
mv wonderland_ch_02_64kb.mp3 $audio_file
rm wonderland_* $zipf
