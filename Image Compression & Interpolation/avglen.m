function res = avglen(sig)
    y=double(unique(sig));
    va = hist(double(sig),y);
    prob= va./sum(sum(va));
    [~,res] = huffmandict(y,prob);
end