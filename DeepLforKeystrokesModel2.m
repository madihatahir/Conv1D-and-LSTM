clc;
clear

for j=1:240
    file=sprintf('%d.xlsx',j);
[num,txt,d]=xlsread(file);

x{j,:}=d;
end
