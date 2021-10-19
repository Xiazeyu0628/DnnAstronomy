function [xsol, rsnr, num] = admm_hybird(y,epsilon,Phit,Phi,Psi,Psit,options,net,im,name)
%Useful functions.
sc = @(z) z*min(epsilon/norm(z(:)), 1); % scaling
% sc为投影函数名 
% 含义：在与0的最大误差为epsilon的解集B中找到一个值u，使其与z的差值最小。
% u = argmin{lB(0，epsilon)(u)+0.5*norm(u-z)^2}

%% Initializations.

% %Dual variable.
 vsol=zeros(size(y));
% 
% %Initial residual/intermediate variable
 s =  - y;
%s = zeros(size(y));

%Initial l2 projection
nsol = sc(s) ;
%nsol = zeros(size(y));

%Creating the initial solution variable with all zeros
xsol = zeros(size(Phi(s)));
%xsol = real((Phi(s)));


num=1;
result=zeros(2,options.max_iter);
noise_matrix = zeros(1,options.max_iter);
image_cell = cell(1,options.max_iter);
f=@(x)  norm(Psit(x),1);
%% loop
while(true) 
xlast=xsol;
% normaliaztion method
% hybird netwok
noise = options.delta*real(Phi(s+nsol-vsol));
temp=(xsol-noise);
noise_energy = norm(noise)
% temp(temp>1)=1;
% temp(temp<0)=0;
xsol = cell2mat(compute_net(net,temp,512));
xsol = xsol/max(max(xsol));
  %在3.13左右的值
xsol(xsol>1)=1;
xsol(xsol<0)=0;
s=Phit(double(xsol))-y;
nsol=sc(vsol-s);
vsol=vsol-(s+nsol);
rsnr = 20*log10(norm(im(:))/norm(im(:)-xsol(:)))
result(1,num) = num;
result(2,num) = rsnr;
% 

if noise_energy>100
   noise_matrix(1,num) = 0;
else
   noise_matrix(1,num) = noise_energy;
end
 
image_cell{num} = xsol;

fval_change = abs(f(xsol)-f(xlast))/f(xsol);
xval_change = norm(xlast-xsol)/norm(xlast);
    
if ((fval_change<options.rel_tol) ||(xval_change<options.rel_tol2)||(num> options.max_iter))
        break;
    end   
num=num+1;
end
% ind = find(noise_matrix(2,:)==0);
% for n = 1:length(ind)
%     noise_matrix(2,n) = mean(noise_matrix(2,ind+1:end));
% end

figure(1)
scatter(result(1,:),result(2,:));
figure_name = ['./project/result/figure/hybird/process/',name,'.jpg'];
saveas(1,figure_name);


figure(2)
scatter(noise_matrix(1,:),noise_matrix(2,:));
figure_name = ['./project/result/figure/hybird/noise/',name,'.jpg'];
saveas(2,figure_name);


[B,I] = sort(result(2,:),'descend');
xsol = image_cell{I(1)};
rsnr = B(1);
num =  result(1,I(1));

close all
end



