function X = sigm(P)
%sigmoid��������Ϊactivation function�����ڼ���activation value
    X = 1./(1+exp(-P));
end