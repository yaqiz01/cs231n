RESULT_PATH=../results

mkdir -p $RESULT_PATH 

#python play.py --mode all --model conv --speedmode 1 --convmode 0 --num-frame 10
#python play.py --mode objdet --path /Users/Yaqi/ee368/kitti/2011_09_26-3/data --start-frame -1 --end-frame 196
#python play.py --mode objdet --path /Users/Yaqi/ee368/kitti/2011_09_26-1/data --start-frame 0 --end-frame 90
#python play.py --mode objdet --path /Users/Yaqi/ee368/kitti/2011_09_26-1/data --start-frame 280 --end-frame -1


#python play.py --mode trainspeed --model conv --speedmode 0 --num-frame 10 &> \
	#$RESULT_PATH/result_$(date +%Y%m%d%H%M%S).txt
#python play.py --mode trainspeed --model conv --convmode 1 --speedmode 0 &> \
	#$RESULT_PATH/result_$(date +%Y%m%d%H%M%S).txt
python play.py --mode trainspeed --model conv --convmode 1 --speedmode 0 --dropout=0.6  --flowmode=2 rseg=50 cseg=100 &> $RESULT_PATH/result_$(date +%Y%m%d%H%M%S).txt &
#python play.py --mode trainspeed --model conv --speedmode 0 --convmode 2 &> \
  #$RESULT_PATH/result_$(date +%Y%m%d%H%M%S).txt
#python play.py --mode trainspeed --model conv --speedmode 2 &> \
  #$RESULT_PATH/result_$(date +%Y%m%d%H%M%S).txt
#python play.py --mode trainspeed --model conv --speedmode 2 --convmode 2 &> \
  #$RESULT_PATH/result_$(date +%Y%m%d%H%M%S).txt
