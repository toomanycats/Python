### script to parse out 'if' statements from C program files ###

input=$1

path=$(echo $input | cut -d, -f 1)
echo $path

#is_c_file=$(echo $path | grep -c -i '*.c')
#
#if [[ $is_c_file -gt 0 ]];then
#    echo $(cat $path | pcregrep -no '(?sm)if\s*\(.*?\)' /dev/null | sed 's/\s//g')
#else
#    echo ""
#fi

exit 0
