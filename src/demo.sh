RESULT_PATH=../results

mkdir -p $RESULT_PATH 

python play.py --mode trainspeed --model conv --speedmode 1 --num-frame 10
#python play.py --mode trainspeed --model conv --speedmode 0 --num-frame 10 &> \
	#$RESULT_PATH/result_$(date +%Y%m%d%H%M%S).txt
