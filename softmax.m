function Y=softmax(W,X,b)
% 函数softmax求输出
%W是输入的系数矩阵，每一行是一个对应类的系数，行数是类别数
%X是数据矩阵，每一列为一个样本的特征向量
%b是偏置
%Y是输出，每一列为对应样本的计算结果，每一列对饮位置的element是对应类的输出计算结果
allcls=exp(W*X+repmat(b,[1 size(X,2)]));
Y=allcls./repmat(sum(allcls),[size(W,1) 1]);

end