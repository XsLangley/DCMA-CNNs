function Y=softmax(W,X,b)
% ����softmax�����
%W�������ϵ������ÿһ����һ����Ӧ���ϵ���������������
%X�����ݾ���ÿһ��Ϊһ����������������
%b��ƫ��
%Y�������ÿһ��Ϊ��Ӧ�����ļ�������ÿһ�ж���λ�õ�element�Ƕ�Ӧ������������
allcls=exp(W*X+repmat(b,[1 size(X,2)]));
Y=allcls./repmat(sum(allcls),[size(W,1) 1]);

end