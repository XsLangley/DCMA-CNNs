function Y=softmax(W,X,b)

allcls=exp(W*X+repmat(b,[1 size(X,2)]));
Y=allcls./repmat(sum(allcls),[size(W,1) 1]);

end