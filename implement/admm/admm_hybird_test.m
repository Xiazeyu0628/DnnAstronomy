function [xsol, rsnr, num,result] = admm_hybird_test(y,epsilon,Phit,Phi,Psi,Psit,options,net,im,name)
%Useful functions.
sc = @(z) z*min(epsilon/norm(z(:)), 1); % scaling


% u = argmin{lB(0，epsilon)(u)+0.5*norm(u-z)^2}

%% Initializations.

 v=zeros(size(y));

 s =  - y;

n = sc(s) ;

xsol = zeros(size(Phi(s)));

% v=zeros(size(y));
% xsol = real(Phi(y));
% xsol = xsol/max(max(xsol));
% s = Phit(xsol) - y;
% n = sc(v-s) ;
% v=v-(s+n);


num=1;
result=zeros(2,options.max_iter);
image_cell = cell(1,options.max_iter);
f=@(x)  norm(Psit(x),1);
%% loop
while(true) 
xlast=xsol;


temp=(xsol-options.delta*real(Phi(s+n-v)));
temp(temp>1)=1;
temp(temp<0)=0;
xsol = cell2mat(compute_net(net,temp,32));
xsol = xsol/max(max(xsol));

%  x_residual = cell2mat(compute_net(net,temp,64));
%  xsol = temp - x_residual;
%  xsol = xsol/max(max(xsol));
 
% 启发于relu
% xsol(xsol>1)=1;
% xsol(xsol<0)=0;  
s=Phit(double(xsol))-y;
n=sc(v-s);
v=v-(s+n);

%记录值
rsnr = 20*log10(norm(im(:))/norm(im(:)-xsol(:)))
result(1,num) = num;
result(2,num) = rsnr;

image_cell{num} = xsol;

fval_change = abs(f(xsol)-f(xlast))/f(xsol);
xval_change = norm(xlast-xsol)/norm(xlast);
    
if ((fval_change<options.rel_tol) ||(xval_change<options.rel_tol2)||(num> options.max_iter))
        break;
    end   
num=num+1
end

[B,I] = sort(result(2,:),'descend');
xsol = image_cell{I(1)};
rsnr = B(1);
num =  result(1,I(1));


end



