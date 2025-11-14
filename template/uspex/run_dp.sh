while :
do
{
  cp /home/cchen/Hiccup/template/uspex/dp_opt.py .
	/home/cchen/apps/deepmd-kit/bin/python dp_opt.py $1

	if [ -f "energy.txt" ]; then
		break
	else
		sleep 0.1
	fi
}
done

