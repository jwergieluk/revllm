#!/bin/bash

pandoc -f markdown -t html5 -s -o index.html revllm_dev.md
pandoc -f markdown -t html5 -s -o imprint.html imprint.md
pandoc -f markdown -t html5 -s -o privacy.html privacy.md

zip -r revllm_dev.zip index.html imprint.html privacy.html *.png
