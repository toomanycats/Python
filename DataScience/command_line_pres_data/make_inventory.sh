#!/bin/bash

function count_lines
{
input=$1
echo $(cat $input | wc -l)
}

function check_for_ascii
{
input=$1
bool=$(file $input | grep -ic "ascii")
if [[ $bool -gt 0 ]];then
    echo 1
else
    echo 0
fi
}

function get_extension
{
input=$1
ext=$(echo $input | rev | cut -d. -f 1 | rev)
echo $ext
}

#### calls here ###
input=$1

count=$(count_lines $input)
ascii_bool=$(check_for_ascii $input)
ext=$(get_extension $input)

if [ $ascii_bool == 1 ];then
    printf "%i,%s\n" $count, $ext
else
    printf "0,%s\n" $ext
fi
