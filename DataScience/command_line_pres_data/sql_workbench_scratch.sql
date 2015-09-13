use LinuxKernel;
SELECT * FROM LinuxKernel.inventory;

alter table inventory add category varchar(20);

select path from inventory where path in (select path_ from sub_list);

update inventory set category = 
       case when inventory.path in 
	     (select path_ from sub_list)
       then "linux" else NULL
 end;

-- join test
select sub_list.path_
from inventory 
left join sub_list
on inventory.path = sub_list.path_;


select * from inventory limit 100;