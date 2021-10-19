function [xsol, rsnr, num] = admm_residual(y,epsilon,Phit,Phi,Psi,Psit,options,net,im)
%Useful functions.
sc = @(z) z*min(epsilon/norm(z(:)), 1); % scaling
% sc为投影函数名 
% 含义：在与0的最大误差为epsilon的解集B中找到一个值u，使其与z的差值最小。
% u = argmin{lB(0，epsilon)(u)+0.5*norm(u-z)^2}



%% Initializations.

%Dual variable.
v=zeros(size(y));

%Initial residual/intermediate variable
s =  - y;

%Initial l2 projection
n = sc(s) ;

%Creating the initial solution variable with all zeros
xsol = zeros(size(Phi(s)));


num=1;
result=zeros(2,options.max_iter);
image_cell = cell(1,options.max_iter);
f=@(x)  norm(Psit(x),1);
%% loop
while(true) 
xlast=xsol;
% normaliaztion method
% hybird netwok

temp=(xsol-options.delta*real(Phi(s+n-v)));

xsol = temp-cell2mat(compute_net(net,temp,32));
xsol = xsol/max(max(xsol));
xsol(xsol>1)=1;
xsol(xsol<0)=0;
s=Phit(double(xsol))-y;
n=sc(v-s);
v=v-(s+n);
rsnr = 20*log10(norm(im(:))/norm(im(:)-xsol(:)));
result(1,num) = num;
result(2,num) = rsnr;
image_cell{num} = xsol;

fval_change = abs(f(xsol)-f(xlast))/f(xsol);
xval_change = norm(xlast-xsol)/norm(xlast);
    if  ((fval_change<options.rel_tol) ||(xval_change<options.rel_tol2)||(num> options.max_iter))
        break;
    end   
num=num+1;
end
scatter(result(1,:),result(2,:));
[B,I] = sort(result(2,:),'descend');
xsol = image_cell{I(1)};
rsnr = B(1);
num =  result(1,I(1));

end



