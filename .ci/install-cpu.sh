# # install python 3.7.16 use pyenv
# sudo apt-get install zlib1g-dev libffi-dev libreadline-dev libssl-dev libsqlite3-dev libncurses5 libncurses5-dev libncursesw5 lzma liblzma-dev libbz2-dev
# pyenv install 3.7.16
# pyenv local 3.7.16

# create and activate virtual environment
if [ ! -d '.venv' ]; then
    python3 -m venv .venv && echo create venv
else
    echo venv exists
fi

source .venv/bin/activate

# # update pip
# python -m pip install -U pip

# # torch cuda 11.3
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# dgl cuda 11.3
# add a source if prefered: -i https://pypi.tuna.tsinghua.edu.cn/simple/
python -m pip install dgl -f https://data.dgl.ai/wheels/cpu/repo.html
python -m pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html

# install requirements
# add a source if prefered: -i https://pypi.tuna.tsinghua.edu.cn/simple/
python -m pip install -r requirements.txt

echo install requirements successfully!
