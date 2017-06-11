input_=$1
file_=$(echo $input_ | cut -d, -f 1 | sed s/\"//g)
out=$(grep -Eo -m 1 "^#include <linux"  ${file_} /dev/null)
echo $out
