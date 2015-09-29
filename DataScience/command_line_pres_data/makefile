root_dir = /home/dcuneo/git/PersonalDS/DataScience/command_line_pres_data
kernel = $(root_dir)/linux-2.6.32.67
inventory = $(root_dir)/linux_inventory.csv
extension_list = $(root_dir)/ext_list.txt
make_inventory = $(root_dir)/make_inventory.sh

define make_ext_list =
cat $(inventory) | cut -d, -f 3 | sort | uniq -c | sort -n > $(extension_list)
endef

define make_inventory =
find $(kernel) -type f | parallel -n 1 --jobs 2 $(make_inventory) > $(inventory)
sed -i '1 i\path_,tot_lines' $(inventory)
endef


$(inventory): $(kernel)
	$(make_ext_list)


$(kernel):
	$(make_inventory)
