#function python {
	#if [[ ! -z "$VIRTUAL_ENV" ]]; then
		#PYTHONHOME=$VIRTUAL_ENV python "$@"
	#else
		#python "$@"
	#fi
#}

#export WORKON_HOME=/home/Yaqi/.virtualenvs
#source /usr/local/bin/virtualenvwrapper.sh
#workon cv

#python play.py --mode all --path /Users/Yaqi/ee368/kitti/2011_09_26-3/data --start-frame -1 --end-frame 196
#python play.py --mode all --path /Users/Yaqi/ee368/kitti/2011_09_26-1/data --start-frame 0 --end-frame 90
#python play.py --mode all --path /Users/Yaqi/ee368/kitti/2011_09_26-1/data --start-frame 280 --end-frame -1

#python play.py --mode trainspeed --model conv --path /Users/Yaqi/ee368/kitti/2011_09_26-3/data --num-frame 10
#python play.py --mode objdet --path /Users/Yaqi/ee368/kitti/2011_09_26-1/data --num-frame 20 --delay 0.5
python play.py --mode trainspeed --model conv --speedmode 2 --num-frame 20
#python play.py --mode trainspeed --model conv --path /Users/Yaqi/ee368/kitti/2011_09_26-1/data --num-frame 10 --delay 0.5
#python play.py --mode trainspeed --model linear --path /Users/Yaqi/ee368/kitti/2011_09_26-1/data --num-frame 10 --delay 0.5
#python play.py --mode all --path /Users/Yaqi/ee368/kitti/2011_09_26-1/data --num-frame 10 --delay 0.5
