function X=flipall(X)
%������תNά����X�ĺ�������X������ά���ϵ����ݽ��з�ת
    for i=1:ndims(X)
        X = flipdim(X,i);
    end
end