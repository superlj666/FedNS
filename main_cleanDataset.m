clear
close all

%load('datasets_clean/a1a_clean.mat'); %rho = 500
% load('datasets_clean/w7a_clean.mat'); %rho = 500
%load('datasets_clean/w8a_clean.mat'); %rho = 500
load('datasets_clean/phishing.mat');
%load('datasets_clean/a9a_clean.mat'); %rho = 500



for j=2:size(X,2)
    temp1=abs(X(:,j));
    temp=max(temp1);
        for i=1:size(X,1)
            %XXXX(i,j)=(XXX(i,j)-mean(XXX(:,j)))/(max(XXX(:,j))-min(XXX(:,j)));
            X(i,j)=X(i,j)/temp;
        end
end




num_feature=size(X,2);
total_sample=size(y,1);
            
%no_workers = 10; %a1a
% no_workers = 80;%60;w7a
%no_workers = 60; %w8a
no_workers = 40; %phishing
%no_workers = 100; %a9a
dataSamples_per_worker=floor(total_sample/no_workers);


total_sample =no_workers*dataSamples_per_worker;

num_iter=50;

lambda_logistic = 1E-3;


% rho=3;   %dataset a1a_clean 
% alpha=0.05;

% rho=100;   %dataset a1a_clean 
% alpha=4;


rho=50;   %dataset w8a_clean w7a 
alpha=1;


X_fede=X;
y_fede=y;

acc = 1E-7;

[obj_snewton]=standard_newton...
    (X_fede,y_fede, no_workers, num_feature, dataSamples_per_worker, num_iter,lambda_logistic);

obj0 = obj_snewton(num_iter);


% num_iter=5000;

num_iter=50;


[obj_znewton, loss_znewton, transmitted_bits_znewton]=newton_zero...
    (X_fede,y_fede, no_workers, num_feature, dataSamples_per_worker, num_iter, obj0, lambda_logistic);


[obj_newton_aadmm, loss_newton_aadmm, transmitted_bits_admm]=newton_ADMM...
    (X_fede,y_fede, no_workers, num_feature, dataSamples_per_worker, num_iter, obj0, lambda_logistic, rho, alpha);




[obj_GD, loss_GD, transmitted_bits_GD]=GD...
    (X_fede,y_fede, no_workers, num_feature, dataSamples_per_worker, num_iter, obj0, lambda_logistic);

bitsToSend=3;
[obj_newton_qadmm, loss_newton_qadmm, transmitted_bits_qadmm]=newton_QADMM...
    (X_fede,y_fede, no_workers, num_feature, dataSamples_per_worker, num_iter, obj0, lambda_logistic,bitsToSend, rho, alpha);


% rho=0.5; %a1a, w8a
% alpha=0.25;

rho=0.1; %phishing
alpha=0.25;

[obj_newton_aadmm_Hk, loss_newton_aadmm_Hk, transmitted_bits_admm_Hk]=newton_ADMM_Hk...
    (X_fede,y_fede, no_workers, num_feature, dataSamples_per_worker, num_iter, obj0, lambda_logistic, rho, alpha);

[obj_newton_qadmm_Hk, loss_newton_qadmm_Hk, transmitted_bits_qadmm_Hk]=newton_QADMM_Hk...
    (X_fede,y_fede, no_workers, num_feature, dataSamples_per_worker, num_iter, obj0, lambda_logistic,bitsToSend, rho, alpha);


% rho=100; %w8a dataset
% alpha=4;

[obj_newton_aadmm_pHk, loss_newton_aadmm_pHk, transmitted_bits_admm_pHk]=newton_ADMM_periodic_Hk...
    (X_fede,y_fede, no_workers, num_feature, dataSamples_per_worker, num_iter, obj0, lambda_logistic, rho, alpha);

[obj_newton_qadmm_pHk, loss_newton_qadmm_pHk, transmitted_bits_qadmm_pHk]=newton_QADMM_periodic_Hk...
    (X_fede,y_fede, no_workers, num_feature, dataSamples_per_worker, num_iter, obj0, lambda_logistic,bitsToSend, rho, alpha);





R = 1;
for i=1:num_iter
    comm_rounds(i) = i*R;
end





h = figure(1);
% subplot(1,2,1)
semilogy(loss_znewton,'k','LineWidth',2);
hold on
semilogy(loss_GD,'LineWidth',2);
semilogy(comm_rounds, loss_newton_aadmm,'r','LineWidth',2);
semilogy(comm_rounds, loss_newton_aadmm_pHk,'b','LineWidth',2);
semilogy(comm_rounds, loss_newton_aadmm_Hk,'m--','LineWidth',2);
semilogy(comm_rounds, loss_newton_qadmm,'b--','LineWidth',2);
semilogy(comm_rounds, loss_newton_qadmm_pHk,'g--','LineWidth',2);
semilogy(comm_rounds, loss_newton_qadmm_Hk,'c--','LineWidth',2);
%semilogy(loss_fedNL,'b','LineWidth',2);
%semilogy(loss_done,'g','LineWidth',2);
xlabel({'Number of communication rounds'},'fontsize',16,'fontname','Times New Roman')
ylabel('Loss','fontsize',16,'fontname','Times New Roman')
legend({'Newton zero','GD', 'Newton-ADMM-r=0','Newton-ADMM-r=0.1', 'Newton-ADMM-r=1'...
    , 'Newton-QADMM-r=0','Newton-QADMM-r=0.1', 'Newton-QADMM-r=1'},'fontsize', 16, 'Location', 'best');
ylim([1E-4 1E4]) 
%xlim([1 1000])
set(gca,'fontsize',14,'fontweight','bold');
print(h, './results/phishing.pdf', '-dpdf','-r600')

% subplot(1,2,2)
% %semilogy(transmitted_bits_signGD,loss_signGD,'k','LineWidth',2);
% semilogy(transmitted_bits_znewton,loss_znewton,'k','LineWidth',2);
% hold on
% semilogy(transmitted_bits_GD,loss_GD,'LineWidth',2);
% semilogy(transmitted_bits_admm,loss_newton_aadmm,'r','LineWidth',2);
% semilogy(transmitted_bits_admm_pHk,loss_newton_aadmm_pHk,'b','LineWidth',2);
% semilogy(transmitted_bits_admm_Hk,loss_newton_aadmm_Hk,'m--','LineWidth',2);
% semilogy(transmitted_bits_qadmm,loss_newton_qadmm,'b--','LineWidth',2);
% semilogy(transmitted_bits_qadmm_pHk, loss_newton_qadmm_pHk,'g--','LineWidth',2);
% semilogy(transmitted_bits_qadmm_Hk,loss_newton_qadmm_Hk,'c--','LineWidth',2);
% %semilogy(transmitted_bits_fedNL,loss_fedNL ,'b','LineWidth',2);

% xlabel({'Communicated bits per node'},'fontsize',16,'fontname','Times New Roman')
% ylabel('Loss','fontsize',16,'fontname','Times New Roman')
% legend(['Newton zero','GD', 'Newton-ADMM-r=0','Newton-ADMM-r=0.1', 'Newton-ADMM-r=1'...
%     , 'Newton-QADMM-r=0','Newton-QADMM-r=0.1', 'Newton-QADMM-r=1'], 'Location', 'best');
% %xlim([1.5E4 1E5])
% %xlim([1E5 6E5])
% ylim([1E-4 1E4]) 
% set(gca,'fontsize',14,'fontweight','bold');

% 
% save results_a1a_clean comm_rounds loss_znewton loss_newton_aadmm  loss_newton_aadmm_Hk...
%     loss_newton_aadmm_pHk loss_newton_qadmm_Hk loss_GD transmitted_bits_znewton...
%     transmitted_bits_GD transmitted_bits_admm transmitted_bits_admm_pHk transmitted_bits_admm_Hk...
%     transmitted_bits_qadmm transmitted_bits_qadmm_pHk transmitted_bits_qadmm_Hk



save results_phishing_clean comm_rounds loss_znewton loss_newton_aadmm  loss_newton_aadmm_Hk...
    loss_newton_aadmm_pHk loss_newton_qadmm_Hk loss_GD transmitted_bits_znewton...
    transmitted_bits_GD transmitted_bits_admm transmitted_bits_admm_pHk transmitted_bits_admm_Hk...
    transmitted_bits_qadmm transmitted_bits_qadmm_pHk transmitted_bits_qadmm_Hk
% 

% save results_w8a_clean comm_rounds loss_znewton loss_newton_aadmm  loss_newton_aadmm_Hk...
%     loss_newton_aadmm_pHk loss_newton_qadmm_Hk loss_GD transmitted_bits_znewton...
%     transmitted_bits_GD transmitted_bits_admm transmitted_bits_admm_pHk transmitted_bits_admm_Hk...
%     transmitted_bits_qadmm transmitted_bits_qadmm_pHk transmitted_bits_qadmm_Hk...
%     loss_newton_qadmm_pHk loss_newton_qadmm

% save results_w7a_clean_v2 comm_rounds loss_znewton loss_newton_aadmm  loss_newton_aadmm_Hk...
%     loss_newton_aadmm_pHk loss_newton_qadmm_Hk loss_GD transmitted_bits_znewton...
%     transmitted_bits_GD transmitted_bits_admm transmitted_bits_admm_pHk transmitted_bits_admm_Hk...
%     transmitted_bits_qadmm transmitted_bits_qadmm_pHk transmitted_bits_qadmm_Hk...
%     loss_newton_qadmm_pHk loss_newton_qadmm

% save results_a9a_clean comm_rounds loss_znewton loss_newton_aadmm  loss_newton_aadmm_Hk...
%     loss_newton_aadmm_pHk loss_newton_qadmm_Hk loss_GD transmitted_bits_znewton...
%     transmitted_bits_GD transmitted_bits_admm transmitted_bits_admm_pHk transmitted_bits_admm_Hk...
%     transmitted_bits_qadmm transmitted_bits_qadmm_pHk transmitted_bits_qadmm_Hk...
%     loss_newton_qadmm_pHk loss_newton_qadmm

