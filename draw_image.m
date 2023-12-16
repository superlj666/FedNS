clear
close all

dataset = 'cod-rna'

load(['./results/', dataset, '_data_sketch_size.mat'], 'sketch_size', 'loss_FedNS', 'loss_FedNDES')

h = figure(1);
semilogy(median(loss_FedNS, 2),'LineWidth',2);
hold on
semilogy(median(loss_FedNDES, 2),'LineWidth',2);
title(dataset)

xticks(1:length(sketch_size))
xticklabels(sketch_size)
xlabel('Sketch size $k$', 'Interpreter','latex', 'fontsize',16,'fontname','Times New Roman')
ylabel('$f(x^t) - f(x^*)$','Interpreter','latex','fontsize', 16, 'fontweight','bold')
legend({'FedNS', 'FedNDES'},'fontsize', 16, 'Location', 'best');
set(gca,'fontsize',14,'fontweight','bold');
print(h, ['./results/', dataset, '_sketch_size.pdf'], '-dpdf','-r600')
