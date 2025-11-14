#!/bin/bash
output_file="/home/cchen/test/ga/ga1/0_O18Cu24/1/wait.txt"
current_dir=$(pwd)

find "$current_dir" -name "POSCAR" -type f | while read poscar_file; do
    echo "$poscar_file" >> "$output_file"
done

while :
do
{
	if [ -f "energy.txt" ]; then
		break
	else
		sleep 0.1
	fi
}
done

