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
    input="$1"
    base=$(basename "$input")
    test_=$(echo "$base" | grep -c "\.")

    if [ $test_ -eq 0 ];then
        echo "NONE"
    else
        ext=$(echo "$base" | rev | cut -d. -f 1 | rev)
    fi

    echo ${ext}
}

#### calls here ###
input=$1

count=$(count_lines $input)
ascii_bool=$(check_for_ascii $input)
ext=$(get_extension $input)

if [ $ascii_bool == 1 ];then
    printf "\"${input}\",\"${count}\",\"${ext}\"\n"
else
    printf "\"{$input}\",\"0\",\"${ext}\"\n"
fi
