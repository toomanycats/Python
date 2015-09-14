#!/bin/bash \n
input=$1 \n
file_=$(echo $input | cut -d, -f 1 | sed s/"//g) \n
out=$(grep -Eo -m 1 "^#include <linux"  "${file_}" /dev/null) \n
echo $out \n
