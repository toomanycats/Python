#!/bin/bash
### script to parse out 'if' statements from C program files ###

echo "$1" | cut -d, -f 1 | xargs -n 1  grep -i "copyright" | sed 's/\W//g' | sed 's/[0-9]//g'

