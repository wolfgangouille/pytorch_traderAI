import torch
import pandas as pd
import os
import ciso8601
import time
import datetime
import numpy

lepath='../python2-krakenex-master/examples/'


#This part lists all files, add leading zeros and sort them by name
zeroes="00000000000"
files=[];
for x in os.listdir(lepath):
    if x.startswith("datakraken"):
        n=x.__len__();
        newx=x[0:10]+zeroes[0:18-n]+x[10::];
        os.rename(lepath+x,lepath+newx)
        files.append(newx);
        print(newx);

files.sort() # sort by name. Maybe sort by date  directly

#This write all file content (without the first and last line)to a single file
for file in files:
    lines = open(lepath+file).readlines()
    fir_lin=0;
    las_lin=0;
    if lines[0]=='M=[\n':
        fir_lin=1;
    if (lines[-1][0]==']' or lines[-1][0]=='\x00'):
        las_lin=1;
    stripped = [line.lstrip() for line in lines[fir_lin:lines.__len__()-las_lin]]
    with open(lepath+'concatenated.txt', 'a') as the_file:
        for line in stripped:
            the_file.write(line);

data = pd.read_csv(lepath+'concatenated.txt',delimiter=' ',header=None)
data=data.to_numpy()
lines = open(lepath+'concatenated.txt').readlines()
new_file = open(lepath+'timestamped_concatenated.txt','a')

for i in range(0,data.__len__()):
    line=data[i,:];
    date=time.mktime(time.struct_time([int(2000+line[2]),int(line[1]),int(line[0]),int(line[3]),int(line[4]),0,0,0,0]));
    new_file.write(str(int(date))+' '+lines[i]);

new_file.close();

#This Reorders the lines
data = pd.read_csv(lepath+'timestamped_concatenated.txt',delimiter=' ',header=None)
data=data.to_numpy()
lines = open(lepath+'timestamped_concatenated.txt').readlines()
sorted_ind=data[:,0].argsort()
data=data[sorted_ind,:];
new_file = open(lepath+'sorted_timestamped_concatenated.txt','a')

for index in sorted_ind:
    new_file.write(lines[index]);

#This detects continuities
discontinuities=[];
date=data[0,0]
for i in range(0,data.__len__()):
    olddate=date;
    date=data[i,0];
    if (date-olddate>120):  #previous_line and current line are contigues (less than 60 sec appart)
        discontinuities.append(i)

#This writes it into  continuous chunks
lines = open(lepath+'sorted_timestamped_concatenated.txt').readlines()
current_file_index=0;
rangestart=0;
for rangeend in discontinuities:
    current_file = open(lepath+'continuous_chunk_'+str(current_file_index)+'.txt','a')
    print('range from '+str(rangestart)+' to '+str(rangeend));
    data=lines[rangestart:rangeend]
    for line in data:
        current_file.write(line);
    rangestart=rangeend;
    current_file_index=current_file_index+1;
    current_file.close()

current_file = open(lepath+'continuous_chunk_'+str(current_file_index)+'.txt','a')
print('range from '+str(rangestart)+' to '+str(lines.__len__()));
data=lines[rangestart:lines.__len__()]
for line in data:
    current_file.write(line);
