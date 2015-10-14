# simple makefile for creating a new inventory and extension list
# if the kernel is updated with new source

# global path prefix
root_dir = /home/dcuneo/git/PersonalDS/DataScience/command_line_pres_data

# prerequistes
kernel = $(root_dir)/linux-2.6.32.67
inventory = $(root_dir)/linux_inventory.csv

# target
extension_list = $(root_dir)/ext_list.txt

# shell script required for recipe
make_inventory = $(root_dir)/make_inventory.sh


$(extension_list): $(inventory)
	cat $(inventory) | cut -d, -f 3 | sort | uniq -c | sort -n > $(extension_list)


$(inventory): $(kernel)
	find $(kernel) -type f | parallel -n 1 --jobs 2 $(make_inventory) > $(inventory)
	sed -i '1 i\path_,tot_lines' $(inventory)
