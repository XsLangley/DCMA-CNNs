function maxindex=FindMaxIndex(mat,scale)
%�ҳ�mat��ÿscale*scale�����е����ֵ��λ�ã�������λ�ø�1������λ�ø�0
%����һ��ȫ��ľ��󣬴�С��matһ��
maxindex=zeros(size(mat));
for l=1:size(mat,3)
    for j=1:scale:size(mat,1)
        for i=1:scale:size(mat,2)
            %�ҵ���l������ĵ�(i,j)����Ӧ��scale��������ֵ�Լ����ֵ�����±꣨rowindex��
            [colmax, rowindex]=max(mat(j:j+scale-1,i:i+scale-1,l));
            %�ҵ������ֵ�����±꣨colindex��
            [~, colindex]=max(colmax);
            %����l�������(i,j)���ڵ����ֵ��Ӧλ��maxindex����1
            maxindex(j-1+rowindex(colindex),i-1+colindex,l)=1;
        end
    end
end
end