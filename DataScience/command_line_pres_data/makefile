root_dir = "/home/dcuneo/git/PersonalDS/DataScience/command_line_pres_data"
kernel = $(root_dir)/"linux-2.6.32.67"
inventory = $(root_dir)/linux_inventory.csv
extension_list = $(root_dir)/ext_list.txt

#function
cat $root_dir/linux_inventory.csv | cut -d, -f 3 | sort | uniq -c | sort -n > $root_dir/ext_list.txt

#function
find $root_dir -type f | parallel -n 1 --jobs 2 run_inventory.sh > $root_dir/linux_inventory.csv
sed -i '1 i\path_,tot_lines' $root_dir/linux_inventory.csv


inventory:
	create_ext_list

kernel:
	run_inventory.sh



