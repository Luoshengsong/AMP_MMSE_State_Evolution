%**************************************************************************
% Based on AMP
%**************************************************************************

clear all; clc; 
close all;
rng('default');

Signal_Model = 'Positive-Bernoulli'; % Gaussian-Bernoulli, Positive-Bernoulli

Phase_random = 'Y'; % Junjie's test

alpha = 0.5; beta = 3;
tanh_denoiser = 'N';
soft_thre = 'Y';

rho = 0.1;          % fraction of nonzeros
lambda = 0.1;       % regularization parameter

N = 2000;     
M = fix(N * 0.6);   % We assumed delta < 1 in the calibration between alpha and threshold

%% Parameters for the Kronecker model
L = ceil( 1/2 + sqrt( 2*M + 1/4 ) );
M = L*(L-1)/2;
delta = M/N;

NSIM = 5;           % number of runs
SNRdB = 100;         % SNR

u_g = 0;
v_g = 1;
% damp = 0.95;     % samll damp is to ensure the convergence of the Kronecker model!!!

Iteration = 50;    % # of iterations of AMP & FISTA
sigma2 = 1/delta * 10^(-SNRdB/10);  

MSE_Gaussian = zeros(Iteration,1);
MSE_Kronecker = zeros(Iteration,1);

if soft_thre == 'Y'
  Monte = 1e8;
  [MSE_SE, SE_tau2] = SE_soft_threshold(Iteration, Monte, rho, v_g, delta, sigma2);
else
  [MSE_SE, SE_tau2] = AMP_SE_MMSE(Signal_Model, Iteration, rho, v_g, sigma2, delta);
end


%% Simulation
for nsim=1:NSIM   
    
    index = random('Binomial',1,rho,N,1);
    if strcmp(Signal_Model,'Gaussian-Bernoulli')
        x = sqrt(v_g) * randn(N,1);
    elseif strcmp(Signal_Model,'Positive-Bernoulli')
        x = sqrt(v_g) * ones(N,1);
    else
        fprintf('no such option\n');
        pause
    end
    x = x .* index;

    %% **********************************************
    %%                  Gaussian model
    %% **********************************************
    
    G = (randn(M,N))/sqrt(M); 
    y_G = G * x + sqrt(sigma2) * randn(M,1); 
    
    % Initialization
    tau2 = zeros(Iteration, 1);
    mse = zeros(Iteration, 1);

    z_G(:,1) = y_G;
    xhat_G(:,1) = zeros(N,1);
    tau2(1) = 1/M*(norm(z_G(:,1),2)^2);
    mse(1) = norm(x - xhat_G(:,1) ,2).^2/N;
    MSE_Gaussian(1) = MSE_Gaussian(1) + mse(1);
    % AMP Iterations
%     damp = 0.97;
    damp = 1;
    for t = 2:Iteration
        r = G' * z_G(:,t-1) + xhat_G(:,t-1);
        % NLE: 
        if strcmp(Signal_Model,'Gaussian-Bernoulli')
            [u_post, v_post] = BG_MMSE_denoiser(r, tau2(t-1), rho, u_g, v_g, N);
        elseif strcmp(Signal_Model,'Positive-Bernoulli')
            if tanh_denoiser == 'Y'
                u_post = sqrt(v_g) * (tanh( beta * (r-alpha) ) + 1) / 2;
                eta_prime = sqrt(v_g) * beta ./ mean( 1 ./ cosh( beta * (r - alpha) ).^2 ) / 2;
            elseif soft_thre == 'Y'
                deniser_parameter = sqrt(rho + rho * tau2(t-1));
                u_post = wthresh(r,'s', deniser_parameter);
                eta_prime = 1/N * nnz(wthresh(r,'s', deniser_parameter));
            else
                [u_post, v_post] = B_MMSE_denoiser(r, tau2(t-1), rho, u_g, v_g);
                eta_prime = v_post / tau2(t-1);
            end
        end
        xhat_G(:,t) = damp * u_post + (1-damp) * xhat_G(:,t-1);
        
        % LE: 
        
        z_G(:,t) = y_G - G * xhat_G(:,t) +  1/delta * z_G(:,t-1) * eta_prime;
    
        tau2(t) = 1/M * (norm(z_G(:,t),2)^2);
        mse(t) = norm(x - xhat_G(:,t),2)^2/N;

        MSE_Gaussian(t) = MSE_Gaussian(t) + mse(t);
    end

    %% **********************************************
    %%                  Kronecker model
    %% **********************************************
    
    L = ceil( 1/2 + sqrt( 2*M + 1/4 ) );
    A = zeros(L*(L-1)/2,N);

    sequence = randn(L,N);
    for n = 1:N
        vector_tmp = sequence(:,n);
        temp = vector_tmp * vector_tmp';
        temp2 = tril(temp,-1); %lower diagonal part
        temp3 = temp2(:);
        A(:,n) = temp3(temp3~=0);
    end
    
    A = A  / sqrt(M);    % normalize
    A = A - mean(A(:));
    
    %% test randomization-Junjie
    if Phase_random == 'Y'
        Phase = sign(randn(M,N));
        A = A .* Phase;
    end
    
    
    y = A * x + sqrt(sigma2) * randn(M,1); 

    % Initialization
    tau2 = zeros(Iteration, 1);
    mse = zeros(Iteration, 1);
    z(:,1) = y;
    xhat_K(:,1) = zeros(N,1);
    tau2(1) = 1/M*(norm(z(:,1),2)^2);
    mse(1) = norm(x - xhat_K(:,1) , 2).^2/N;
    MSE_Kronecker(1) = MSE_Kronecker(1) + mse(1);
    % AMP Iterations
    damp = 0.6;
%     damp = 1;
    for t = 2: Iteration
        r = A'*z(:,t-1) + xhat_K(:,t-1);
        % NLE: 
        if strcmp(Signal_Model,'Gaussian-Bernoulli')
            [u_post, v_post] = BG_MMSE_denoiser(r, tau2(t-1), rho, u_g, v_g, N);
        elseif strcmp(Signal_Model,'Positive-Bernoulli')
            if tanh_denoiser == 'Y'
                u_post = sqrt(v_g) * (tanh( beta * (r-alpha) ) + 1) / 2;
                eta_prime = sqrt(v_g) * beta ./ mean( 1 ./ cosh( beta * (r - alpha) ).^2 ) / 2;
            elseif soft_thre == 'Y'
                deniser_parameter = sqrt(rho + rho * tau2(t-1));
                u_post = wthresh(r,'s', deniser_parameter);
                eta_prime = 1/N * nnz(wthresh(r,'s', deniser_parameter));
            else
                [u_post, v_post] = B_MMSE_denoiser(r, tau2(t-1), rho, u_g, v_g);
                eta_prime = v_post / tau2(t-1);
            end
        end
        xhat_K(:,t) = damp * u_post + (1-damp) * xhat_K(:, t-1);

        % LE: 
%         eta_prime = v_post / tau2(t-1);
        z(:,t) = y - A*xhat_K(:,t) + 1/delta * z(:,t-1) * eta_prime;

        tau2(t) = 1/M*(norm(z(:,t),2)^2);
        mse(t) = norm(x - xhat_K(:,t),2)^2/N;
   
        MSE_Kronecker(t) = MSE_Kronecker(t) + mse(t);

    end
    
    %% display
    if(mod(nsim,1)==0)
        fprintf('************************************************************\n');
        fprintf('snr = %d dB, nsim = %d\n',SNRdB,nsim);
        fprintf('MSE_Gaussian = %e, MSE_Kronecker = %e \n',MSE_Gaussian(Iteration)/nsim,MSE_Kronecker(Iteration)/nsim);
    end

end

MSE_Gaussian = MSE_Gaussian/NSIM;
MSE_Kronecker = MSE_Kronecker/NSIM;

figure;
semilogy(... 1:Iteration,MSE_SE,'k-.v', ...
         1:Iteration, MSE_Gaussian,'r-x', ...
         1:Iteration,MSE_Kronecker,'g-o', ...
         'LineWidth',1.5 ...
         );
legend('State Eolution', 'Gaussian model: AMP','Kronecker model: AMP')
ylabel('MSE');
% axis([1 Iteration sigma2 1]);
title('LASSO with MMSE denoiser');
grid on;
