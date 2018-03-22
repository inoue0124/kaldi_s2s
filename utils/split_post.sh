#!/bin/bash

csplit -k -n 3 -f utt post.1 /^phrase*/ {$1}
