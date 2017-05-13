function python {
	if [[ ! -z "$VIRTUAL_ENV" ]]; then
		PYTHONHOME=$VIRTUAL_ENV /opt/local/bin/python "$@"
	else
		/opt/local/bin/python "$@"
	fi
}

python play.py --mode all --path /Users/Yaqi/ee368/kitti/2011_09_26-3/data --start-frame -1 --end-frame 196
python play.py --mode all --path /Users/Yaqi/ee368/kitti/2011_09_26-1/data --start-frame 0 --end-frame 90
python play.py --mode all --path /Users/Yaqi/ee368/kitti/2011_09_26-1/data --start-frame 280 --end-frame -1
