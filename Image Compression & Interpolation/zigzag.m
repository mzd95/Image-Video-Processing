function res = zigzag(x)
h = 1; v = 1;
vmin = 1; hmin = 1;

vmax = size(x, 1);
hmax = size(x, 2);

i = 1;
res = zeros(1, vmax * hmax);
while ((v <= vmax) && (h <= hmax))
    if (mod(h + v, 2) == 0)
        if (v == vmin)       
            res(i) = x(v, h); 
            if (h == hmax)
                v = v + 1;
            else
                h = h + 1;
            end
            i = i + 1;
        elseif ((h == hmax) && (v < vmax))
            res(i) = x(v, h);
            v = v + 1;
            i = i + 1;
        elseif ((v > vmin) && (h < hmax))
            res(i) = x(v, h);
            v = v - 1;
            h = h + 1;
            i = i + 1;
        end
    else
        if ((v == vmax) && (h <= hmax))
            res(i) = x(v, h);
            h = h + 1;
            i = i + 1;
        elseif (h == hmin)
            res(i) = x(v, h);
            if (v == vmax)
                h = h + 1;
            else
                v = v + 1;
            end
            i = i + 1;
        elseif ((v < vmax) && (h > hmin))
            res(i) = x(v, h);
            v = v + 1;
            h = h - 1;
            i = i + 1;
        end
    end
    if ((v == vmax) && (h == hmax))
        res(i) = x(v, h);
        break
    end
end