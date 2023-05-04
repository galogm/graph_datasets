# # install python 3.9.16 on ubuntu 18.04, check /usr/local or /usr/bin for openssl. If none, installation is needed.
# cat <<"EOF" | bash
# sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev tk-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev -y && \
# curl -O https://www.python.org/ftp/python/3.9.16/Python-3.9.16.tar.xz && \
# tar -xf Python-3.9.16.tar.xz && \
# cd Python-3.9.16 && \
# ./configure --enable-optimizations --with-openssl=/usr && \
# make -j 4 && \
# sudo make altinstall && \
# python3.9 --version
# EOF

# create and activate virtual environment
python3.9 -m venv .venv
source .venv/bin/activate

# update pip
python -m pip install -U pip

# install requirements
python -m pip install -r requirements.txt

pre-commit install

echo install requirements successfully!
