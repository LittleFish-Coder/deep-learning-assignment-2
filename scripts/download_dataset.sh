# This script downloads the dataset from the link and unzips it into the dataset directory.
echo "Downloading the dataset..."

# create a directory to store the dataset
mkdir -p dataset

# get the dataset from the link
wget https://cchsu.info/files/images.zip

# unzip the dataset into the directory
unzip images.zip -d dataset