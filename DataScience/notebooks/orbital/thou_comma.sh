### print thousands separator ###

alias printf=/usr/bin/printf
input=$1
read input
printf "%'d\n"  $input
