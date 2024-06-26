rm -rf .env

bash .ci/py.sh

# create and activate virtual environment
python3 -m venv .env
source .env/bin/activate

# update pip
# add a source if prefered: -i https://pypi.tuna.tsinghua.edu.cn/simple/
python -m pip install -U pip

# install requirements
# add a source if prefered: -i https://pypi.tuna.tsinghua.edu.cn/simple/
python -m pip install -r requirements-dev.txt

pre-commit install

echo install DEV requirements successfully!
