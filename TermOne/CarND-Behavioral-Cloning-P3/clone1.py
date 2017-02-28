import csv
import cv2
import numpy as np

lines=[]
with open('../data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    print (line)
    lines.append(line)

for line in lines:
  source_path = line[0]
  filename = source_path.split('\\')[-1]
  print (filename)




















