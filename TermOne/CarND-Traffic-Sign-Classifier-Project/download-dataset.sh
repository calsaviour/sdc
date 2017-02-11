#!/bin/sh

dataset_dir="traffic-signs-data"

if [ -d "$dataset_dir" ]
then
	echo "Downloaded dataset"
else
	echo "Dataset not not found"
	echo "Downloading data set"
        wget https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
	echo "Download complete"

	echo "Unzip dataset"
	unzip traffic-signs-data.zip -d traffic-signs-data
fi

