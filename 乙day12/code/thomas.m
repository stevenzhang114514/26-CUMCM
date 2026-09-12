function x = thomas(dl, dd, du, b)
    n  = numel(b);
    cp = zeros(n,1); dp = zeros(n,1);
    cp(1) = du(1)/dd(1);  dp(1) = b(1)/dd(1);
    for i = 2:n
        m = dd(i) - dl(i)*cp(i-1);
        if i < n, cp(i) = du(i)/m; end
        dp(i) = (b(i) - dl(i)*dp(i-1))/m;
    end
    x = zeros(n,1);  x(n) = dp(n);
    for i = n-1:-1:1
        x(i) = dp(i) - cp(i)*x(i+1);
    end
end