function res = izigzag(x,vmax,hmax)
h = 1; v = 1;
vmin = 1; hmin = 1;
res = zeros(vmax, hmax);
i = 1;

while ((v <= vmax) && (h <= hmax))
    if (mod(h + v, 2) == 0)
        if (v == vmin)
            res(v, h) = x(i);
            if (h == hmax)
                v = v + 1;
            else
                h = h + 1;
            end
            i = i + 1;
        elseif ((h == hmax) && (v < vmax))
            res(v, h) = x(i);
            i;
            v = v + 1;
            i = i + 1;
        elseif ((v > vmin) && (h < hmax))
            res(v, h) = x(i);
            v = v - 1;
            h = h + 1;
            i = i + 1;
        end
    else
        if ((v == vmax) && (h <= hmax))
            res(v, h) = x(i);
            h = h + 1;
            i = i + 1;
        elseif (h == hmin)
            res(v, h) = x(i);
            if (v == vmax)
                h = h + 1;
            else
                v = v + 1;
            end
            i = i + 1;
        elseif ((v < vmax) && (h > hmin))
            res(v, h) = x(i);
            v = v + 1;
            h = h - 1;
            i = i + 1;
        end
    end
    if ((v == vmax) && (h == hmax))
        res(v, h) = x(i);
        break;
    end
end