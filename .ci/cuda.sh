# update key for ubuntu 18.04
# sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# clean cuda
sudo apt clean
sudo apt update
sudo apt purge cuda
sudo apt purge nvidia-*
sudo apt autoremove

# intall cuda
# check version on https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=deb_network
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-11-3

echo install cuda successfully!
