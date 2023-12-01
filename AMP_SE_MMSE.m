function [SE_MSE, SE_tau2] = AMP_SE_MMSE(Signal_Model, Iteration, rho, v_g, sigma2, delta)

    if strcmp(Signal_Model,'Gaussian-Bernoulli')

        nu = v_g;
        lim = inf;
        
        SE_MSE = zeros(Iteration, 1);
        SE_tau2 = zeros(Iteration, 1);
        
        SE_MSE(1) = rho * nu;
        SE_tau2(1) = sigma2 + 1/delta * SE_MSE(1);
        bound = 200;
        for it = 2: Iteration
            tau = SE_tau2( it - 1 );
            f = @(r) r.*r ./ ...
                (    1   +   (1-rho)./rho .* sqrt( 1 + nu./tau )   .*   exp( max(-bound, -0.5 * r .* r .* nu./ tau./(nu + tau) ) )     ) ...
                .* normpdf(r, 0, sqrt(nu + tau))  ;
        
            SE_MSE(it) = rho * nu - rho * nu^2 / (nu + tau)^2 * integral(f,-lim,lim);
            SE_tau2(it) = sigma2 + 1/delta * SE_MSE(it);
        end
    
    elseif strcmp(Signal_Model,'Positive-Bernoulli')

        theta = sqrt(v_g);
        lim = inf;
        
        SE_MSE = zeros(Iteration, 1);
        SE_tau2 = zeros(Iteration, 1);
        
        SE_MSE(1) = rho * v_g;
        SE_tau2(1) = sigma2 + 1/delta * SE_MSE(1);
        
        bound = 200;
        fb = @(b) (b > bound) .* bound + (b < -bound) .* (-bound) + (abs(b) <= bound) .* b;
        for it = 2: Iteration
            tau = SE_tau2( it - 1 );
            f = @(r) 1 ./ ...
                ( 2 + (1-rho)./rho .* exp( fb(-0.5 * ( 2 * theta .* r - theta .* theta) / tau ) )  + rho./(1-rho) .* exp( fb(0.5 * ( 2 * theta .* r - theta .* theta) / tau  )) ) ...
                .* ( (1-rho) .* normpdf(r, 0, sqrt(tau)) +  rho .* normpdf(r, theta, sqrt(tau)) );
        
            SE_MSE(it) = theta^2 * integral(f,-lim,lim);
            SE_tau2(it) = sigma2 + 1/delta * SE_MSE(it);
        end
    
    else
        fprintf('no such option\n');
        pause
    end

end