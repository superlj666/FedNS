function [obj_FedNDES, loss_FedNDES, transmitted_bits]=FedNDES...
    (XX,YY, no_workers, num_feature, noSamples, num_iter, obj0, lambda_logistic, m, method, mu)

    s1=num_feature;
    s2=noSamples;
    grads=ones(num_feature,no_workers);
    out_central=zeros(s1,1);

    max_iter = num_iter;
    a = 0.1
    b = 0.5
    max_iter_inner = 50
    delta = 0.3 / no_workers

    for i = 1:max_iter
        transmitted_bits(i) = i*num_feature*32;
        for ii =1:no_workers
            first = (ii-1)*s2+1;
            last = first+s2-1;
            
            S(:, :, ii) = gen_sketch_mat(m, num_feature, method);
            grads(:,ii)=-(XX(first:last,1:num_feature)'*(YY(first:last)./(1+exp(YY(first:last).*(XX(first:last,1:num_feature)*out_central)))))+lambda_logistic*out_central;
    
            temp = lambda_logistic*eye(num_feature,num_feature);
            for jj=first:last
                temp=temp+YY(jj)^2*XX(jj,:)'*XX(jj,:)*(exp(YY(jj)*XX(jj,:)*out_central)/(1+exp(YY(jj)*XX(jj,:)*out_central))^2);
            end
            sketchHessian(:,:,ii) = S(:, :, ii) * sqrtm(temp);
            hessian(:, :, ii) = sketchHessian(:,:,ii)' * sketchHessian(:,:,ii);
        end
    
        Hessian = sum(hessian,3)/no_workers;
        gradient = sum(grads,2)/no_workers;
        Newton_step= - Hessian \ gradient;
        lambda_w = gradient' * Newton_step;
        if lambda_w > 3/4 * delta
            ii = randperm(no_workers, 1)
            first = (ii-1)*s2+1;
            last = first+s2-1;

            mu = 1;

            iter_num = 0;
            while sum(log(1+exp(-YY(first:last).*(XX(first:last,1:s1)*(out_central + mu*Newton_step))))) > sum(log(1+exp(-YY(first:last).*(XX(first:last,1:s1)*out_central)))) + a * mu * lambda_w + 1e-13 && iter_num < max_iter_inner
                mu = b * mu;
                iter_num = iter_num + 1
            end
        end

        out_central = out_central + mu*Newton_step;
        final_obj =lambda_logistic*0.5*norm(out_central)^2;
        for ii =1:no_workers
            first = (ii-1)*s2+1;
            last = first+s2-1;
            %final_obj = final_obj + 0.5*norm(XX(first:last,1:s1)*out_central - YY(first:last))^2;
            final_obj = final_obj+sum(log(1+exp(-YY(first:last).*(XX(first:last,1:s1)*out_central))));
        end
        %i
        obj_FedNDES(i)=final_obj;
        
        final_obj
        abs(final_obj-obj0)        
        
        loss_FedNDES(i)=abs(final_obj-obj0);
    end  
end
     
