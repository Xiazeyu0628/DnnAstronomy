function x = HS_adjoint_operator(y, Gw, At,N, M)

if iscell(y)
    c = length(y);
    x = zeros(N, M, c);

    %
    for ind = 1:c
        x(:,:,ind) = At(Gw{ind}' * y{ind});
    end
else
    x(:,:,1) = At(Gw{1}' * y);
end

end

