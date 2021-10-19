function [xsol, fval, t] = admm_contvdn(y,epsilon,Phit,Phi,options,paramtv)
% ADMM based algorithm to solve the following problem:
%
%   min ||x||_tv   s.t.  ||y - Phit x||_2 < epsilon,
%
% which is posed as the equivalent problem:
%
%   min ||x||_tv   
%
%   s.t.  n = y - Phit x and ||n||_2 < epsilon.
%
% Inputs:
%
% - y is the measurement vector. 
% - epsilon is a bound on the residual error.
% - A is the forward measurement operator and At the associated adjoint 
%   operator.
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
%   - rho: penalty parameter for constraint violation (default: 1e2).
%
%   - delta: step size for the proximal gradient update (default: 1e0).
%
%     To guarantee convergence, the following condition is needed:
%
%        delta*L < 2
%
%        where L is the square norm of the operator Phit, i.e. the square
%        of its maximum singular value.
%
% - paramtv is Matlab structure that sets the parameters for the proxtv
%   function. It contains the following fields:
%
%   - verbose: 0 no log, 1 print main steps, 2 print all steps.
%
%   - rel_obj: minimum relative change of the proximal function value 
%
%   - max_iter: max. number of iterations.
% 
% 
% Outputs:
%
% - xsol: solution of the problem.
%
% - fval : objective value.
%
% - t : number of iterations used by the algorithm.
%
% Note: In order to compute the TV prox function you can use
%       the prox_tv function included in the src folder.
%       Prox TV usage:
%
%         [sol, tvnorm] = prox_tv(b, lambda, paramtv)
% 
%       Inputs:
% 
%         - b: input 2D signal.
%         - lambda: constant for the TV proximal function.
%         - param: MATLAB structure that sets the parameters for the TV proximal function.
% 
%       Outputs:
% 
%         -sol: solution of the TV proximal function
%         -tvnorm: TV norm of the solution.
%
 
 
%Useful functions.
sc = @(z) z*min(epsilon/norm(z(:)), 1); % scaling
 
 
%%Initializations.
 
%Dual variable.
v=zeros(size(y));
 
%Initial solution (all zero solution)
%Initial residual/intermediate variable
s =  - y;
 
%Initial l2 projection
n = sc(s) ;
 
%Creating the initial solution variable
%with all zeros
xsol = zeros(size(Phi(s)));

t=0;
%% Main loop. 
temp=xsol-options.delta*(real(Phi(s+n-v))/max(max(real(Phi(s+n-v)))));
%grad=real(Phi(Phit(xsol)-y));
%temp=xsol-options.delta*grad;
%Write your code here
 while(true) 
[xsol, tvnorm1] = prox_tv(temp, options.delta/options.rho, paramtv);
xsol = xsol/max(max(xsol));
s=Phit(xsol)-y;
n=sc(v-s);
v=v-(s+n);
%grad=real(Phi(Phit(xsol)-y));
%temp=xsol-options.delta*grad;
temp=xsol-options.delta*(real(Phi(s+n-v))/max(max(real(Phi(s+n-v)))));
xlast=xsol;
[xsol, tvnorm2] = prox_tv(temp, options.delta/options.rho, paramtv);
xsol = xsol/max(max(xsol));
t=t+1;
%fval=tvnorm2-tvnorm1;
(abs(fval)/tvnorm2)
    if (  ( (abs(fval)/tvnorm2<options.rel_tol)&&(norm(y-Phit(xsol))<=epsilon) )||((norm(xsol-xlast)/norm(xsol))<options.rel_tol2)||(t>=options.max_iter)   )
        break;
    end
    
end
 
end