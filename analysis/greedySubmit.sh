c=0
while [[ $c -le 13 ]]
do
   nohup python2.7 greedyPicking.py $c & 
   echo "nohup python2.7 greedyPicking.py $c &"
   let c=c+1
done
