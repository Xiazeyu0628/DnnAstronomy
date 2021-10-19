function sparseness = sparseness(X)
% X可以是向量或是矩阵
[m,n] = size(X);
num = m*n;

% 分别计算L1和L2范数
s1 = norm(X,1);
s2 = norm(X,2);

% 计算稀疏度
s2 = sqrt(s2);
c = s1/s2;
a = sqrt(num)-c;
b = sqrt(num)-1;
sparseness = a/b;
end