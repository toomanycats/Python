root_dir="/home/daniel/git/Python2.7/DataScience/command_line_pres_data"

input_=$1
file_=$(echo $input_ | cut -d, -f 1 | sed s/\"//g)
out=$(grep -Eo -m 1 "^#include <linux"  "${root_dir}/${file_}" /dev/null)
echo $out
