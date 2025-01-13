% Advanced DSGE Model with Impulse Response Functions (IRFs)
% Author: Adelia Nur Fasha
% Objective: Extend DSGE simulation with Impulse Response Functions (IRFs)
% to observe the dynamic response of risk variables to economic shocks.

clc; clear; close all;

%% Simulation Settings
numSimulations = 10000;  % Number of Monte Carlo simulations
T = 40;                  % Time horizon for IRFs (40 periods)

%% DSGE Parameters
rho_A = 0.9;  % Productivity shock persistence
sigma_A = 0.02;  % Volatility of productivity shock

rho_m = 0.85;  % Monetary policy persistence
sigma_m = 0.01;  % Volatility of monetary policy shock

rho_f = 0.8;   % Fiscal policy persistence
sigma_f = 0.015;  % Volatility of fiscal policy shock

%% Risk Parameters (Aligned with Existing Model)
EAD = 1e6;               % Exposure at Default ($1 million)
LGD = 0.45;              % Loss Given Default (45%)
PD = 0.02;               % Probability of Default (2%)
marketVolatility = 0.015; % 1.5% daily volatility
marketExposure = 2e6;     % $2 million investment
opIncidentRate = 0.03;    % 3% chance of failure
opImpact = 5e5;           % $500,000 impact
complianceViolationRate = 0.01; % 1% violation probability
compliancePenalty = 1e5;        % $100,000 penalty

%% Initialize Shock Vectors
A_shock = zeros(T,1);  % Productivity Shock
m_shock = zeros(T,1);  % Monetary Policy Shock
f_shock = zeros(T,1);  % Fiscal Policy Shock

% Apply a one-time shock at t=1
A_shock(1) = sigma_A;  % Productivity shock
m_shock(1) = sigma_m;  % Monetary policy shock
f_shock(1) = sigma_f;  % Fiscal policy shock

%% Initialize Risk Responses (IRFs)
creditRisk_IRF = zeros(T,1);
marketRisk_IRF = zeros(T,1);
operationalRisk_IRF = zeros(T,1);
complianceRisk_IRF = zeros(T,1);
totalRisk_IRF = zeros(T,1);

%% IRF Simulation
for t = 2:T
    % Shock propagation (AR(1) process)
    A_shock(t) = rho_A * A_shock(t-1);
    m_shock(t) = rho_m * m_shock(t-1);
    f_shock(t) = rho_f * f_shock(t-1);
    
    % Credit Risk Response
    creditRisk_IRF(t) = PD * EAD * LGD * (1 + A_shock(t));
    
    % Market Risk Response
    marketRisk_IRF(t) = abs(marketVolatility * marketExposure * (1 + m_shock(t)));
    
    % Operational Risk Response
    operationalRisk_IRF(t) = opIncidentRate * opImpact * (1 + f_shock(t));
    
    % Compliance Risk Response
    complianceRisk_IRF(t) = complianceViolationRate * compliancePenalty * (1 + f_shock(t));
    
    % Total Risk
    totalRisk_IRF(t) = creditRisk_IRF(t) + marketRisk_IRF(t) + operationalRisk_IRF(t) + complianceRisk_IRF(t);
end

%% Visualization - Impulse Response Functions
figure;
plot(1:T, creditRisk_IRF, 'b-', 'LineWidth', 2); hold on;
plot(1:T, marketRisk_IRF, 'r--', 'LineWidth', 2);
plot(1:T, operationalRisk_IRF, 'g-.', 'LineWidth', 2);
plot(1:T, complianceRisk_IRF, 'm:', 'LineWidth', 2);
plot(1:T, totalRisk_IRF, 'k-', 'LineWidth', 2);
xlabel('Time (Periods)');
ylabel('Risk Exposure ($)');
title('Impulse Response Functions (IRFs) to Economic Shocks');
legend('Credit Risk', 'Market Risk', 'Operational Risk', 'Compliance Risk', 'Total Risk');
grid on;

%% Results Summary
fprintf('--- IRF Analysis Summary ---\n');
fprintf('Max Credit Risk Impact: $%.2f\n', max(creditRisk_IRF));
fprintf('Max Market Risk Impact: $%.2f\n', max(marketRisk_IRF));
fprintf('Max Operational Risk Impact: $%.2f\n', max(operationalRisk_IRF));
fprintf('Max Compliance Risk Impact: $%.2f\n', max(complianceRisk_IRF));
fprintf('Max Total Risk Impact: $%.2f\n', max(totalRisk_IRF));
% Gertler-Kiyotaki (GK) Model - Basic Financial Friction DSGE
% Define Parameters

beta = 0.99;      % Discount factor
sigma = 1;        % Risk aversion
phi = 1.5;        % Taylor rule inflation response
theta = 0.75;     % Price stickiness (Calvo parameter)
alpha = 0.33;     % Capital share in production
delta = 0.025;    % Capital depreciation rate

% Banking Sector Parameters
phi_b = 0.92;     % Fraction of banker net worth retained
eta = 0.25;       % Bank leverage ratio
zeta = 0.10;      % Fraction of bankers surviving

% Shock Parameters
rho_a = 0.9;      % Persistence of productivity shock
sigma_a = 0.01;   % Std. dev. of productivity shock
% Initialize Variables
T = 100; % Simulation horizon
A = zeros(T,1); % Productivity shock
K = zeros(T,1); % Capital stock
N_b = zeros(T,1); % Banker net worth
L = zeros(T,1); % Loans
Y = zeros(T,1); % Output
R = zeros(T,1); % Interest rate
pi = zeros(T,1); % Inflation

% Initial Conditions
A(1) = 0; K(1) = 1; N_b(1) = 0.1; L(1) = 0.5; Y(1) = K(1)^alpha; R(1) = 1;

% Productivity Shock (AR(1))
for t = 2:T
    A(t) = rho_a * A(t-1) + sigma_a * randn;
end

% Model Dynamics
for t = 2:T
    % Output
    Y(t) = A(t) * K(t-1)^alpha;
    
    % Bank Leverage Constraint: Loans limited by bank net worth
    L(t) = eta * N_b(t-1);
    
    % Banker Net Worth Evolution
    N_b(t) = phi_b * ((R(t-1) - 1) * L(t-1)) + zeta * N_b(t-1);
    
    % Capital Accumulation
    K(t) = (1 - delta) * K(t-1) + L(t);
    
    % Taylor Rule for Interest Rate
    pi(t) = (Y(t) - Y(t-1)) / Y(t-1); % Inflation approximation
    R(t) = 1 + phi * pi(t);
end
% Plot Results
figure;

subplot(2,2,1);
plot(Y, 'LineWidth', 2);
title('Output (Y)');
xlabel('Time');
ylabel('Output');

subplot(2,2,2);
plot(N_b, 'LineWidth', 2);
title('Bank Net Worth (N_b)');
xlabel('Time');
ylabel('Net Worth');

subplot(2,2,3);
plot(L, 'LineWidth', 2);
title('Loans (L)');
xlabel('Time');
ylabel('Loans');

subplot(2,2,4);
plot(R, 'LineWidth', 2);
title('Interest Rate (R)');
xlabel('Time');
ylabel('Interest Rate');

sgtitle('Gertler-Kiyotaki (GK) Model Dynamics');
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
alpha = 0.33;           % Capital share in production
eta = 0.5;              % Capital adjustment cost

% Financial Friction Parameters
phi_k = 0.36;           % Leverage ratio (fraction of capital funded by debt)
kappa = 0.1;            % Cost of external financing

%% Steady State Values
k_ss = ((alpha / (1 / beta - 1 + delta)))^(1 / (1 - alpha));
y_ss = k_ss^alpha;
investment_ss = delta * k_ss;

%% Simulation Setup
T = 100;                 % Time periods
shock = sigma_e * randn(T, 1); % Technology shock

% Initialize variables
capital = zeros(T, 1);
output = zeros(T, 1);
investment = zeros(T, 1);
bank_debt = zeros(T, 1);
credit_spread = zeros(T, 1);
firm_net_worth = zeros(T, 1);
interest_rate = zeros(T, 1);

capital(1) = k_ss;
output(1) = y_ss;
investment(1) = investment_ss;
bank_debt(1) = phi_k * k_ss;
firm_net_worth(1) = k_ss - bank_debt(1);
interest_rate(1) = 0.05;
credit_spread(1) = 0.02;

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
    
    % Firm net worth
    firm_net_worth(t) = capital(t) - bank_debt(t);
    
    % Credit spread dynamics
    credit_spread(t) = kappa * (bank_debt(t) / firm_net_worth(t));
    
    % Interest rate response (Taylor Rule)
    interest_rate(t) = max(0, 0.05 + phi * (output(t) - y_ss));
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

grid on;

figure;
subplot(2,2,1);
plot(firm_net_worth, 'LineWidth', 1.5);
title('Firm Net Worth (N_f)');
ylabel('Net Worth'); xlabel('Time'); grid on;

subplot(2,2,2);
plot(credit_spread, 'LineWidth', 1.5);
title('Credit Spread');
ylabel('Spread'); xlabel('Time'); grid on;

subplot(2,2,3);
plot(interest_rate, 'LineWidth', 1.5);
title('Interest Rate (R)');
ylabel('Rate'); xlabel('Time'); grid on;

subplot(2,2,4);
plot(output, 'LineWidth', 1.5);
title('Output (Y)');
ylabel('Output'); xlabel('Time'); grid on;

%% Interpretation
% - Observe how shocks propagate through output, investment, and bank debt.
% - Higher leverage increases the economy's sensitivity to financial shocks.
% - Bank debt amplifies business cycle fluctuations.
% - Credit spread and firm net worth dynamics provide additional insights into financial frictions.
