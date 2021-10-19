function [xsol, rsnr, num,result] = admm_hybird_convergence(y,~,Phit,Phi,Psi,Psit,options,net,im)
%Useful functions.
sc = @(z) z*min(epsilon/norm(z(:)), 1); % scaling
%% Initializations.


vsol=zeros(size(y));
s =  - y;
nsol = (options.rho/(2+options.rho))*(vsol-s) ;
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

shrink=@(z,lambda) max(abs(z)-lambda,0).*sign(z);

f=@(x)  options.lambda*norm(Psit(x),1)+ norm(y - Phit(x))^2;
% Distance_sol = 0;
%% loop
while(true) 
xlast=xsol;
% vlast = vsol;
% nlast = nsol;
% normaliaztion method
% hybird netwok
%fidelity_gradient = real(Phi(s+nsol-vsol));
% xsol=Psi(shrink(Psit(xsol-options.delta*fidelity_gradient),options.rho^(-1)*options.delta*options.lambda));
% 
temp=(xsol-options.delta*real(Phi(s+nsol-vsol)));
temp(temp>1)=1;
temp(temp<0)=0;
x_residual = cell2mat(compute_net(net,temp,512)); 
xsol = temp - x_residual;
xsol = xsol/max(max(xsol));
% xsol(xsol>1)=1;
% xsol(xsol<0)=0;
xsol =double(xsol);
s=Phit(xsol)-y;

nsol=(options.rho/(2+options.rho))*(vsol-s);

vsol=vsol-(s+nsol);
rsnr = 20*log10(norm(im(:))/norm(im(:)-xsol(:)))
result(1,num) = num;
result(2,num) = rsnr;
image_cell{num} = xsol;

    
fval_change = abs(f(xsol)-f(xlast))/f(xsol);
xval_change = norm(xlast-xsol)/norm(xlast);
    if  ((fval_change<options.rel_tol) ||(xval_change<options.rel_tol2)||(num> options.max_iter))
    %if((num> options.max_iter)&&(rsnr>0))
        break;
    end   
num=num+1;
end
[B,I] = sort(result(2,:),'descend');
xsol = image_cell{I(1)};

rsnr = B(1);
num =  result(1,I(1));

end



