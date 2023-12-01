function [MSE_SE, tau2_SE] = SE_soft_threshold(Iteration, Monte, rho, v_g, delta, sigma2)

MSE_SE  = zeros(Iteration, 1);
tau2_SE = zeros(Iteration, 1);

MSE_SE(1) = rho * v_g;
tau2_SE(1) = sigma2 + 1 / delta * MSE_SE(1);

for it = 2: Iteration
    tau2 = tau2_SE(it-1);
    deniser_parameter = sqrt(rho+rho*tau2);
    x = sqrt(v_g) * ones(Monte, 1);
    index = random('Binomial', 1, rho, Monte, 1);
    x = x .* index;
    r = x + sqrt(tau2) * randn(Monte, 1);
    MSE_SE(it) = mean( (wthresh(r, 's', deniser_parameter) - x).^2 );
    tau2_SE(it) = sigma2 + 1 / delta * MSE_SE(it);
end


end