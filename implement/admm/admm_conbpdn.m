function [xsol, fval, t] = admm_conbpdn(y,epsilon,Phit,Phi,Psi,Psit,options)
% ADMM based algorithm to solve the following problem:
%
%   min ||Psit x||_1   s.t.  ||y - Phit x||_2 < epsilon,
%
% which is posed as the equivalent problem:
%
%   min ||Psit x||_1   
%
%   s.t.  n = y - Phit x and ||n||_2 < epsilon.
%
% Inputs:
%
% - y is the measurement vector. 
% - epsilon is a bound on the residual error.
% - Phit is the forward measurement operator and Phi the associated adjoint 
%   operator.
% - Psit is a sparfying transform and Psi its adjoint.
% - options is a Matlab structure containing the following fields:
% 
%   - verbose: 0 no log, 1 print main steps, 2 print all steps.
%
%   - rel_tol: minimum relative change of the objective value (default:
%     1e-4). The algorithm stops if
%
%           | f( x(t) ) - f( x(t-1) ) | | / | f( x(t) ) | < rel_tol
%     and   ||y - Phit x(t)||_2 < epsilon,
%
%     where x(t) is the estimate of the solution at iteration t.
%
%   - rel_tol2: second condition for stopping the algorithm. rel_tol2 is 
%     the minimum relative change of the iterates (default:
%     1e-4). The algorithm stops if
%           || x(t) - x(t-1) ||_2 | / || x(t) ||_2 < rel_tol2
%
%     where x(t) is the estimate of the solution at iteration t.
%
%   - max_iter: max. number of iterations (default: 200).
%
%   - rho: penalty parameter for ADMM (default: 1e2).
%
%   - delta: step size for the proximal gradient update (default: 1e0).
%
%   To guarantee convergence, the following condition is needed:
%
%      delta*L <= 1
%
%      where L is the square norm of the operator Phit, i.e. the square of
%      its maximum singular value.
% 
% Outputs:
%
% - xsol: solution of the problem.
%
% - fval : objective value.
%
% - t : number of iterations used by the algorithm.




% Optional input arguments.
if ~isfield(options, 'verbose'), options.verbose = 1; end
if ~isfield(options, 'rel_tol'), options.rel_tol = 1e-4; end
if ~isfield(options, 'rel_tol2'), options.rel_tol2 = 1e-4; end
if ~isfield(options, 'max_iter'), options.max_iter = 200; end
if ~isfield(options, 'rho'), options.rho = 1e2; end
if ~isfield(options, 'delta'), options.delta = 1; end

%Useful functions.
sc = @(z) z*min(epsilon/norm(z(:)), 1); % scaling
% sc为投影函数名 
% 含义：在与0的最大误差为epsilon的解集B中找到一个值u，使其与z的差值最小。
% u = argmin{lB(0，epsilon)(u)+0.5*norm(u-z)^2}

%%Initializations.

%Dual variable.
v=zeros(size(y));

%Initial residual/intermediate variable
s =  - y;

%Initial l2 projection
n = sc(s) ;

%Creating the initial solution variable with all zeros
xsol = zeros(size(Phi(s)));


t=0;
f=@(x)  norm(Psit(x),1);
shrink=@(z,lambda) max(abs(z)-lambda,0).*sign(z);
%% Main loop. 

%Write your code here
while(true) 
xlast=xsol;
xsol=Psi(shrink(Psit(xsol-options.delta*real(Phi(s+n-v))),options.rho^(-1)*options.delta));
s=Phit(xsol)-y;
n=sc(v-s);
v=v-(s+n);
t=t+1;
fval=f(xsol)-f(xlast);
%object_change=abs(fval)/f(xsol)
%relative_change=norm(y-Phit(xsol))
    if (  ( (abs(fval)/f(xsol)<options.rel_tol)&&(norm(y-Phit(xsol))<=epsilon) )||((norm(xlast-xsol)/norm(xlast))<options.rel_tol2))
        break;
    end
end
end

