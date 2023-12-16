clear
close all

dataset = 'cod-rna'
load(['datasets_clean/', dataset]);

for j=2:size(X,2)
    temp1=abs(X(:,j));
    temp=max(temp1);
        for i=1:size(X,1)
            X(i,j)=X(i,j)/temp;
        end
end

num_feature=size(X,2);
total_sample=size(y,1);

if dataset == "covtype"
    mu = 1;
    no_workers = 200;
    rho=50;
    alpha=1;
    k = 20;    
    num_iter = 10;
    sketch_size = 5:2:23;
elseif dataset == "SUSY" 
    mu = 1;
    no_workers = 60;
    rho=50;
    alpha=1;
    k = 20;
    num_iter = 15;
    sketch_size = 3:3:30;
elseif dataset == "cod-rna" 
    mu = 1;
    no_workers = 60;
    rho=50;
    alpha=1;
    k = 15;
    num_iter = 15;
    sketch_size = 3:3:30;

elseif dataset == "phishing" % iter = 11
    mu = 1;
    no_workers = 40; 
    rho=0.1; 
    alpha=0.25;
    k = 17;    
    num_iter = 10;
    sketch_size = 10:5:50;
end

lambda_logistic = 1E-3;
repeat = 1;

dataSamples_per_worker=floor(total_sample/no_workers);

total_sample =no_workers*dataSamples_per_worker;

X_fede=X;
y_fede=y;

[loss_snewton]=standard_newton...
    (X_fede,y_fede, no_workers, num_feature, dataSamples_per_worker, num_iter,lambda_logistic);
obj0 = loss_snewton(end);


for j = 1:length(sketch_size)
    for i = 1:repeat
        k = sketch_size(j);
        [~, loss, ~]=FedNS...
            (X_fede,y_fede, no_workers, num_feature, dataSamples_per_worker, num_iter, obj0, lambda_logistic, k, "Gaussian", mu);
        loss_FedNS(j, i) = loss(end);
        [~, loss, ~]=FedNDES...
            (X_fede,y_fede, no_workers, num_feature, dataSamples_per_worker, num_iter, obj0, lambda_logistic, k+5, "Gaussian", mu);
        loss_FedNDES(j, i) = loss(end);
    end
end

save(['./results/', dataset, '_data_sketch_size.mat'], 'sketch_size', 'loss_FedNS', 'loss_FedNDES')
