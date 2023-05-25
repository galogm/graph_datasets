IP=$1
PORT=$2
nohup jupyter notebook --ip=${IP:=10.214.199.218} --port=${PORT:=8888} --no-browser --config=./configs/nb.py > logs/nb.log &
# http://10.214.199.218:8888/?token=1e982906ea039d74580bb72d323ab38e67d6c1ae704c8215
