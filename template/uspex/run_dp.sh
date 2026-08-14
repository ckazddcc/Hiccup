while :
do
{
  cp <YOUR_TEMPLATE_DIR>/uspex/dp_opt.py .
	<YOUR_PYTHON_PATH> dp_opt.py $1

	if [ -f "energy.txt" ]; then
		break
	else
		sleep 0.1
	fi
}
done

