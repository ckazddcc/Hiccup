#!/bin/bash
while :
do
{
  cp <YOUR_TEMPLATE_DIR>/uspex/mace_opt.py .
	<YOUR_PYTHON_PATH> mace_opt.py $1

	if [ -f "energy.txt" ]; then
		break
	else
		sleep 0.1
	fi
}
done

