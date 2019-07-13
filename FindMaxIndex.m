function maxindex=FindMaxIndex(mat,scale)
%找出mat中每scale*scale区块中的最大值的位置，并将该位置赋1，其他位置赋0
%建立一个全零的矩阵，大小与mat一样
maxindex=zeros(size(mat));
for l=1:size(mat,3)
    for j=1:scale:size(mat,1)
        for i=1:scale:size(mat,2)
            %找到第l个矩阵的第(i,j)所对应的scale块的列最大值以及最大值的行下标（rowindex）
            [colmax, rowindex]=max(mat(j:j+scale-1,i:i+scale-1,l));
            %找到列最大值的列下标（colindex）
            [~, colindex]=max(colmax);
            %将第l个矩阵的(i,j)块内的最大值对应位在maxindex上置1
            maxindex(j-1+rowindex(colindex),i-1+colindex,l)=1;
        end
    end
end
end