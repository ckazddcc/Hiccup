#!/bin/bash
while :
do
{
  cp /home/chencheng/Hiccup/template/uspex/mace_opt.py .
	/home/cchen/apps/deepmd-kit/bin/python mace_opt.py $1

	if [ -f "energy.txt" ]; then
		break
	else
		sleep 0.1
	fi
}
done

