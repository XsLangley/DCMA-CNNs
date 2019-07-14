function maxindex=FindMaxIndex(mat,scale)

maxindex=zeros(size(mat));
for l=1:size(mat,3)
    for j=1:scale:size(mat,1)
        for i=1:scale:size(mat,2)
            
            [colmax, rowindex]=max(mat(j:j+scale-1,i:i+scale-1,l));
            
            [~, colindex]=max(colmax);
            
            maxindex(j-1+rowindex(colindex),i-1+colindex,l)=1;
        end
    end
end
end