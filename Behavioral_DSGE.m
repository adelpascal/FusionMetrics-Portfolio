% Financial Friction DSGE Model (Gertler-Kiyotaki/BGG Framework)

% Clear workspace and define parameters
clear; clc; close all;

%% Model Parameters
beta = 0.99;            % Discount factor
sigma = 1;              % Risk aversion
phi = 1;                % Inverse Frisch elasticity
rho = 0.9;              % Persistence of technology shock
sigma_e = 0.01;         % Std. deviation of technology shock
delta = 0.025;          % Capital depreciation rate
alpha = 0.33;          % Capital share in production
eta = 0.5;              % Capital adjustment cost

% Financial Friction Parameters
phi_k = 0.36;           % Leverage ratio (fraction of capital funded by debt)
kappa = 0.1;            % Cost of external financing

%% Steady State Values
k_ss = ((alpha / (1 / beta - 1 + delta)))^(1 / (1 - alpha));
y_ss = k_ss^alpha;
investment_ss = delta * k_ss;
interest_rate_ss = 1 / beta - 1;
credit_spread_ss = kappa * phi_k;

%% Simulation Setup
T = 100;                 % Time periods
shock = sigma_e * randn(T, 1); % Technology shock

% Initialize variables
capital = zeros(T, 1);
output = zeros(T, 1);
investment = zeros(T, 1);
bank_debt = zeros(T, 1);
interest_rate = zeros(T, 1);
credit_spread = zeros(T, 1);

capital(1) = k_ss;
output(1) = y_ss;
investment(1) = investment_ss;
bank_debt(1) = phi_k * k_ss;
interest_rate(1) = interest_rate_ss;
credit_spread(1) = credit_spread_ss;

%% Model Simulation
for t = 2:T
    % Productivity shock
    z_t = rho * shock(t - 1) + shock(t);
    
    % Output with financial frictions
    output(t) = exp(z_t) * capital(t - 1)^alpha;
    
    % Investment decision with adjustment costs
    investment(t) = ((1 - kappa) * beta * output(t)) / (1 + eta);
    
    % Capital accumulation
    capital(t) = (1 - delta) * capital(t - 1) + investment(t);
    
    % Bank debt evolution
    bank_debt(t) = phi_k * capital(t);
    
    % Interest rate dynamics (Taylor Rule)
    interest_rate(t) = interest_rate_ss + phi * (output(t) - y_ss);
    
    % Credit spread dynamics
    credit_spread(t) = kappa * (bank_debt(t) / capital(t));
end

%% Plot Results
figure;
subplot(3,1,1);
plot(output, 'LineWidth', 1.5);
title('Output Dynamics with Financial Frictions');
ylabel('Output'); grid on;

subplot(3,1,2);
plot(investment, 'LineWidth', 1.5);
title('Investment Dynamics');
ylabel('Investment'); grid on;

subplot(3,1,3);
plot(bank_debt, 'LineWidth', 1.5);
title('Bank Debt Evolution');
ylabel('Debt'); xlabel('Time'); grid on;

figure;
subplot(2,1,1);
plot(interest_rate, 'LineWidth', 1.5);
title('Interest Rate Dynamics');
ylabel('Interest Rate'); grid on;

subplot(2,1,2);
plot(credit_spread, 'LineWidth', 1.5);
title('Credit Spread Dynamics');
ylabel('Credit Spread'); xlabel('Time'); grid on;

grid on;

%% Interpretation
% - Observe how shocks propagate through output, investment, bank debt, interest rates, and credit spreads.
% - Higher leverage increases the economy's sensitivity to financial shocks.
% - Bank debt and credit spreads amplify business cycle fluctuations.
% Financial Friction DSGE Model (Gertler-Kiyotaki/BGG Framework)

% Clear workspace and define parameters
clear; clc; close all;

%% Model Parameters
beta = 0.99;            % Discount factor
sigma = 1;              % Risk aversion
phi = 1;                % Inverse Frisch elasticity
rho = 0.9;              % Persistence of technology shock
sigma_e = 0.05;         % Increased Std. deviation of technology shock (Amplified Shock)
delta = 0.025;          % Capital depreciation rate
alpha = 0.33;           % Capital share in production
eta = 0.5;              % Capital adjustment cost

% Financial Friction Parameters
phi_k = 0.5;            % Increased Leverage ratio (fraction of capital funded by debt)
kappa = 0.2;            % Increased Cost of external financing

%% Steady State Values
k_ss = ((alpha / (1 / beta - 1 + delta)))^(1 / (1 - alpha));
y_ss = k_ss^alpha;
investment_ss = delta * k_ss;
interest_rate_ss = 1 / beta - 1;
credit_spread_ss = kappa * phi_k;

%% Simulation Setup
T = 100;                 % Time periods
shock = sigma_e * randn(T, 1); % Stronger Technology shock

% Initialize variables
capital = zeros(T, 1);
output = zeros(T, 1);
investment = zeros(T, 1);
bank_debt = zeros(T, 1);
interest_rate = zeros(T, 1);
credit_spread = zeros(T, 1);

capital(1) = k_ss;
output(1) = y_ss;
investment(1) = investment_ss;
bank_debt(1) = phi_k * k_ss;
interest_rate(1) = interest_rate_ss;
credit_spread(1) = credit_spread_ss;

%% Model Simulation
for t = 2:T
    % Productivity shock
    z_t = rho * shock(t - 1) + shock(t);
    
    % Output with financial frictions
    output(t) = exp(z_t) * capital(t - 1)^alpha;
    
    % Investment decision with adjustment costs
    investment(t) = ((1 - kappa) * beta * output(t)) / (1 + eta);
    
    % Capital accumulation
    capital(t) = (1 - delta) * capital(t - 1) + investment(t);
    
    % Bank debt evolution
    bank_debt(t) = phi_k * capital(t);
    
    % Interest rate dynamics (Taylor Rule)
    interest_rate(t) = interest_rate_ss + phi * (output(t) - y_ss);
    
    % Credit spread dynamics with financial shock
    credit_spread(t) = kappa * (bank_debt(t) / capital(t)) + 0.1 * randn();
end

%% Plot Results
figure;
subplot(3,1,1);
plot(output, 'LineWidth', 1.5);
title('Output Dynamics with Financial Frictions');
ylabel('Output'); grid on;

subplot(3,1,2);
plot(investment, 'LineWidth', 1.5);
title('Investment Dynamics');
ylabel('Investment'); grid on;

subplot(3,1,3);
plot(bank_debt, 'LineWidth', 1.5);
title('Bank Debt Evolution');
ylabel('Debt'); xlabel('Time'); grid on;

figure;
subplot(2,1,1);
plot(interest_rate, 'LineWidth', 1.5);
title('Interest Rate Dynamics');
ylabel('Interest Rate'); grid on;

subplot(2,1,2);
plot(credit_spread, 'LineWidth', 1.5);
title('Credit Spread Dynamics with Financial Shock');
ylabel('Credit Spread'); xlabel('Time'); grid on;

grid on;

%% Interpretation
% - Amplified shocks now generate stronger responses across output, investment, bank debt, interest rates, and credit spreads.
% - Increased leverage and financing costs intensify economic sensitivity to financial disturbances.
% - Credit spreads now exhibit more dynamic behavior due to the introduced financial shocks.
