tbl = readtable("AORD_2012_2022.xlsx");

% Centering the returns

returns = tbl.Return - mean(tbl.Return);
y = returns;

%RWMH 
dim = 3; % specify dimension of theta
Niter = 2650; % We want 20000 iterations
markov_chain = zeros(Niter,dim); % Create a vector to store the chain
sig2 = zeros(Niter,1);
sig2(1) = var(y);


theta0 = [0.4347 0.5128 0.02677];
theta0 = randn(1,dim);% Starting value. Using a standard normal random number for simplicity
markov_chain(1,:) = theta0; % Just setting starting value as the first value in the markov chain
n = 1;



while n<Niter
    if n>1
        t1 = markov_chain(n-1,1);
        t2 = markov_chain(n-1,2);
        t3 = markov_chain(n-1,3);

        omega = exp(t1);
        alpha = (exp(t2)*exp(t3))/(1+exp(t2)+exp(t3)+exp(t2)*exp(t3));
        beta = (exp(t2))/(1+exp(t2)+exp(t3)+exp(t2)*exp(t3));

        sig2(n) = omega + alpha*y(n-1)^2 + beta*sig2(n-1);
    else
        ;
    end
    
    Sigma = eye(dim)*0.001;

    epsilon = mvnrnd(zeros(dim,1),Sigma); % Drawing from proposal dist
    proposal = markov_chain(n,:)+epsilon; % Adding epsilon to get the new proposal
    auxiliary = kernel(proposal,y(n),sig2(n))-kernel(markov_chain(n,:),y(n),sig2(n)); % Computing our diff in log-likelihoods for proposal and previous value
    alpha = min(exp(auxiliary),1); % Computing acceptance prob. Remembering to exp the auxillary as it was in log-scale
    u = rand; % Generate U from uniform. We will use to accept/reject our new proposa
    % Simple if-else, for accepting/rejecting our proposal. If
    % U<acceptance prob, then accept, else reject.
    if u<alpha 
        markov_chain(n+1,:) = proposal;
    else
        markov_chain(n+1,:) = markov_chain(n,:);
    end
    n = n+1;
end

%% Creating our figure of plots
figure
hold on

subplot(3,1,1)
plot(markov_chain(:,1))
title('\theta_1')

subplot(3,1,2)
plot(markov_chain(:,2))
title('\theta_2')

subplot(3,1,3)
plot(markov_chain(:,3))
title('\theta_3')

%%

%https://www.oreilly.com/library/view/machine-learning-for/9781492085249/ch04.html#callout_machine_learning_based___span_class__keep_together__volatility_prediction__span__CO3-5
%%
function llh = kernel(theta,y,sig2)

omega = @(theta) exp(theta(1));
alpha = @(theta) (exp(theta(2))*exp(theta(3)))/(1+exp(theta(2))+exp(theta(3))+exp(theta(2))*exp(theta(3)));
beta = @(theta) (exp(theta(2)))/(1+exp(theta(2))+exp(theta(3))+exp(theta(2))*exp(theta(3)));

% Formula of log-likelihood
log_like = @(y, sig2) -1/2*sum(log(2*pi)+log(sig2)+(y^2/sig2^2));

% Log prior
gamma_constant = gamma(11.5)/(gamma(1.5)*gamma(10));

log_prior = @(alpha, beta) log(gamma_constant*(alpha^0.5)*((1-alpha)^9)) + log(gamma_constant*(beta^9)*((1-beta)^0.5));

% Jacobian

detJ = @(theta) -((exp(theta(1)+2*theta(2)+theta(3)))/((exp(theta(2)+1))^3)*(exp(theta(3)+1))^2);

% Calculating the kernel: log prior + log likelihood + log detJ
llh = log_prior(theta(2), theta(3))+log_like(y, sig2)+log(detJ(theta));
end