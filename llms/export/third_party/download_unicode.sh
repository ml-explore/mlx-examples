#!/bin/bash

commit=1204f9727005974587d6fc1dcd4d4f0ead87c856
url=https://raw.githubusercontent.com/ggerganov/llama.cpp/${commit}/src/

for file in 'unicode.cpp' 'unicode.h' 'unicode-data.cpp' 'unicode-data.h'
do
  curl -OL ${url}/${file}
done

touch unicode_downloaded
