function X = sigm(P)
%sigmoid函数；作为activation function，用于计算activation value
    X = 1./(1+exp(-P));
end