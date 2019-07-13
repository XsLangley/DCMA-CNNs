function X=flipall(X)
%用来翻转N维矩阵X的函数，将X在所有维度上的数据进行翻转
    for i=1:ndims(X)
        X = flipdim(X,i);
    end
end