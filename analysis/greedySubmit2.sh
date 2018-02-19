c=15
while [[ $c -le 30 ]]
do
   nohup python2.7 greedyPicking.py $c & 
   echo "nohup python2.7 greedyPicking.py $c &"
   let c=c+1
done
