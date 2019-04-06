function res = rlc(sig)
s = []; len = [];
s(1) = sig(1);
len(1) = 1;
ind = 1;
for i = 2:length(sig)
    if sig(i - 1) == sig(i)
        len(ind) = len(ind) + 1;
    else
        ind = ind + 1;
        s(ind) = sig(i);
        len(ind) = 1;
    end
end

R = 8;
Rl = floor(len/R);
mRl = max(Rl);    
if mRl > 0
    dl = len-R*floor(len/R);
    rl = zeros(mRl+1,length(len));
    rl(mRl+1,:) = dl;
    for k=1:mRl
        rl(k,:) = R*(Rl>=k);
    end
    S = s(ones(mRl+1,1),:);
    len = rl(:)';
    s = S(:)';
    lnz = len>0;
    len = len(lnz);
    s = s(lnz);
end

for i = 1:length(len)
    res(i*2 - 1) = len(i);
    res(i*2) = s(i);
end