function res = derlc(sig)
c = size(sig);
if c(1) == 1
    m = 1;
    for n = 1:2:length(sig)
        for k = m:m + sig(n) - 1
            res(k) = sig(n+1);
        end
        m = m + sig(n);
    end
else
    1
    len = c(1);
    m = 1; n = 1;
    while(n <= len)
        m = 1;
        for k = 1:2:length(sig(n,:))
            if(sig(n,k) == 0)
                n = n + 1;
            else
                for k2 = m:m + sig(n,k) - 1
                    res(n,k) = sig(n,k+1);
                end
                m = m + sig(n,k);
            end
        end
        n = n + 1;
    end
end