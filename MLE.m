rng(2020) 
data = xlsread('AORD_2012_2022.xlsx'); % load the data
returns = data - mean(data);
y = returns;

Mdl = garch(1,1);

estimate(Mdl, y)
