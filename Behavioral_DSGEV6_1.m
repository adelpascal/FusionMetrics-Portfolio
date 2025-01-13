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
% Behavioral DSGE Model with Global Economic Shocks

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

% Behavioral Bias Parameters
lambda_myopia = 0.1;    % Degree of myopia
lambda_overconfidence = 0.1; % Degree of overconfidence

% Global Economic Shock Parameters
gamma_g = 0.2;          % Sensitivity to global shocks
rho_g = 0.8;            % Persistence of global economic shock
sigma_g = 0.02;         % Std. deviation of global economic shock

%% Steady State Values
k_ss = ((alpha / (1 / beta - 1 + delta)))^(1 / (1 - alpha));
y_ss = k_ss^alpha;
investment_ss = delta * k_ss;
interest_rate_ss = 1 / beta - 1;

%% Simulation Setup
T = 100;                      % Time periods
tech_shock = sigma_e * randn(T, 1); % Technology shock
global_shock = sigma_g * randn(T, 1); % Global economic shock

% Initialize variables
capital = zeros(T, 1);
output = zeros(T, 1);
investment = zeros(T, 1);
interest_rate = zeros(T, 1);

capital(1) = k_ss;
output(1) = y_ss;
investment(1) = investment_ss;
interest_rate(1) = interest_rate_ss;

%% Model Simulation
for t = 2:T
    % Global economic shock propagation
    g_t = rho_g * global_shock(t - 1) + global_shock(t);
    
    % Adjust output with global shock and behavioral biases
    output(t) = exp(tech_shock(t) + gamma_g * g_t) * (capital(t - 1)^alpha);
    
    % Behavioral adjustment to investment (myopia)
    investment(t) = ((1 - lambda_myopia) * beta * output(t)) / (1 + eta);
    
    % Capital accumulation
    capital(t) = (1 - delta) * capital(t - 1) + investment(t);
    
    % Interest rate dynamics with overconfidence behavior
    interest_rate(t) = interest_rate_ss + phi * ((1 + lambda_overconfidence) * (output(t) - y_ss));
end

%% Plot Results
figure;
subplot(3,1,1);
plot(output, 'LineWidth', 1.5);
title('Output Dynamics with Global Economic Shocks');
ylabel('Output'); grid on;

subplot(3,1,2);
plot(investment, 'LineWidth', 1.5);
title('Investment Dynamics with Behavioral Bias');
ylabel('Investment'); grid on;

subplot(3,1,3);
plot(interest_rate, 'LineWidth', 1.5);
title('Interest Rate Dynamics with Overconfidence');
ylabel('Interest Rate'); xlabel('Time'); grid on;

grid on;

%% Interpretation
% - Global shocks amplify output volatility.
% - Myopia reduces investment responsiveness.
% - Overconfidence leads to exaggerated interest rate movements.
% Behavioral DSGE Model with Global Economic Shocks and Macroprudential Policies

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

% Behavioral Bias Parameters
lambda_myopia = 0.1;    % Degree of myopia
lambda_overconfidence = 0.1; % Degree of overconfidence

% Global Economic Shock Parameters
gamma_g = 0.2;          % Sensitivity to global shocks
rho_g = 0.8;            % Persistence of global economic shock
sigma_g = 0.02;         % Std. deviation of global economic shock

% Macroprudential Policy Parameters
ccb_sensitivity = 0.3;  % Countercyclical capital buffer sensitivity
ltv_limit = 0.8;        % Loan-to-value ratio limit

%% Steady State Values
k_ss = ((alpha / (1 / beta - 1 + delta)))^(1 / (1 - alpha));
y_ss = k_ss^alpha;
investment_ss = delta * k_ss;
interest_rate_ss = 1 / beta - 1;

%% Simulation Setup
T = 100;                      % Time periods
tech_shock = sigma_e * randn(T, 1); % Technology shock
global_shock = sigma_g * randn(T, 1); % Global economic shock

% Initialize variables
capital = zeros(T, 1);
output = zeros(T, 1);
investment = zeros(T, 1);
interest_rate = zeros(T, 1);
ccb = zeros(T, 1); % Countercyclical capital buffer
ltv_ratio = zeros(T, 1); % Loan-to-value ratio

capital(1) = k_ss;
output(1) = y_ss;
investment(1) = investment_ss;
interest_rate(1) = interest_rate_ss;
ccb(1) = 0.02; % Initial capital buffer
ltv_ratio(1) = ltv_limit;

%% Model Simulation
for t = 2:T
    % Global economic shock propagation
    g_t = rho_g * global_shock(t - 1) + global_shock(t);
    
    % Adjust output with global shock and behavioral biases
    output(t) = exp(tech_shock(t) + gamma_g * g_t) * (capital(t - 1)^alpha);
    
    % Countercyclical capital buffer increases during booms
    ccb(t) = max(0.02, ccb_sensitivity * (output(t) - y_ss));
    
    % Loan-to-value ratio adjusts with output fluctuations
    ltv_ratio(t) = max(0.6, ltv_limit - 0.1 * (output(t) - y_ss));
    
    % Behavioral adjustment to investment (myopia) with policy impact
    investment(t) = ((1 - lambda_myopia) * beta * output(t)) / (1 + eta + ccb(t));
    
    % Capital accumulation
    capital(t) = (1 - delta) * capital(t - 1) + investment(t);
    
    % Interest rate dynamics with overconfidence behavior and CCB impact
    interest_rate(t) = interest_rate_ss + phi * ((1 + lambda_overconfidence) * (output(t) - y_ss)) + ccb(t);
end

%% Plot Results
figure;
subplot(4,1,1);
plot(output, 'LineWidth', 1.5);
title('Output Dynamics with Global Economic Shocks and Macroprudential Policies');
ylabel('Output'); grid on;

subplot(4,1,2);
plot(investment, 'LineWidth', 1.5);
title('Investment Dynamics with Behavioral Bias and CCB');
ylabel('Investment'); grid on;

subplot(4,1,3);
plot(interest_rate, 'LineWidth', 1.5);
title('Interest Rate Dynamics with Overconfidence and CCB');
ylabel('Interest Rate'); xlabel('Time'); grid on;

subplot(4,1,4);
plot(ccb, 'LineWidth', 1.5);
title('Countercyclical Capital Buffer Dynamics');
ylabel('CCB'); xlabel('Time'); grid on;

%% Interpretation
% - Global shocks amplify output volatility.
% - Myopia reduces investment responsiveness.
% - Overconfidence leads to exaggerated interest rate movements.
% - Macroprudential tools (CCB and LTV) stabilize investment and credit growth.
% Behavioral DSGE Model with Global Economic Shocks, Macroprudential Policies, and MPLB

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

% Behavioral Bias Parameters
lambda_myopia = 0.1;    % Degree of myopia
lambda_overconfidence = 0.1; % Degree of overconfidence

% Global Economic Shock Parameters
gamma_g = 0.2;          % Sensitivity to global shocks
rho_g = 0.8;            % Persistence of global economic shock
sigma_g = 0.02;         % Std. deviation of global economic shock

% Macroprudential Policy Parameters
ccb_sensitivity = 0.3;  % Countercyclical capital buffer sensitivity
ltv_limit = 0.8;        % Loan-to-value ratio limit
mplb_sensitivity = 0.25; % MPLB sensitivity to economic conditions

%% Steady State Values
k_ss = ((alpha / (1 / beta - 1 + delta)))^(1 / (1 - alpha));
y_ss = k_ss^alpha;
investment_ss = delta * k_ss;
interest_rate_ss = 1 / beta - 1;

%% Simulation Setup
T = 100;                      % Time periods
tech_shock = sigma_e * randn(T, 1); % Technology shock
global_shock = sigma_g * randn(T, 1); % Global economic shock

% Initialize variables
capital = zeros(T, 1);
output = zeros(T, 1);
investment = zeros(T, 1);
interest_rate = zeros(T, 1);
ccb = zeros(T, 1); % Countercyclical capital buffer
mplb = zeros(T, 1); % Macroprudential Liquidity Buffer
ltv_ratio = zeros(T, 1); % Loan-to-value ratio

capital(1) = k_ss;
output(1) = y_ss;
investment(1) = investment_ss;
interest_rate(1) = interest_rate_ss;
ccb(1) = 0.02; % Initial capital buffer
mplb(1) = 0.05; % Initial liquidity buffer
ltv_ratio(1) = ltv_limit;

%% Model Simulation
for t = 2:T
    % Global economic shock propagation
    g_t = rho_g * global_shock(t - 1) + global_shock(t);

    % Output with global shock and behavioral biases
    output(t) = exp(tech_shock(t) + gamma_g * g_t) * (capital(t - 1)^alpha);

    % Countercyclical capital buffer increases during booms
    ccb(t) = max(0.02, ccb_sensitivity * (output(t) - y_ss));

    % MPLB tightens during booms, loosens during downturns
    mplb(t) = max(0.03, mplb_sensitivity * (output(t) - y_ss));

    % Loan-to-value ratio adjusts with output fluctuations
    ltv_ratio(t) = max(0.6, ltv_limit - 0.1 * (output(t) - y_ss));

    % Investment with MPLB and behavioral bias
    investment(t) = ((1 - lambda_myopia) * beta * output(t)) / (1 + eta + ccb(t) + mplb(t));

    % Capital accumulation
    capital(t) = (1 - delta) * capital(t - 1) + investment(t);

    % Interest rate dynamics with overconfidence, CCB, and MPLB
    interest_rate(t) = interest_rate_ss + phi * ((1 + lambda_overconfidence) * (output(t) - y_ss)) + ccb(t) + mplb(t);
end

%% Plot Results
figure;
subplot(5,1,1);
plot(output, 'LineWidth', 1.5);
title('Output Dynamics with Global Shocks, CCB, and MPLB');
ylabel('Output'); grid on;

subplot(5,1,2);
plot(investment, 'LineWidth', 1.5);
title('Investment with Behavioral Bias, CCB, and MPLB');
ylabel('Investment'); grid on;

subplot(5,1,3);
plot(interest_rate, 'LineWidth', 1.5);
title('Interest Rate with Overconfidence, CCB, and MPLB');
ylabel('Interest Rate'); xlabel('Time'); grid on;

subplot(5,1,4);
plot(ccb, 'LineWidth', 1.5);
title('Countercyclical Capital Buffer (CCB)');
ylabel('CCB'); xlabel('Time'); grid on;

subplot(5,1,5);
plot(mplb, 'LineWidth', 1.5);
title('Macroprudential Liquidity Buffer (MPLB)');
ylabel('MPLB'); xlabel('Time'); grid on;

%% Interpretation
% - Global shocks amplify output volatility.
% - Myopia reduces investment responsiveness.
% - Overconfidence leads to exaggerated interest rate movements.
% - CCB stabilizes credit growth; MPLB stabilizes liquidity.
%% Open Economy DSGE with Behavioral Biases
clear; clc; close all;

%% Parameters
beta = 0.99;        % Discount factor
sigma = 1;          % Risk aversion
phi = 1.5;          % Taylor rule inflation response
theta = 0.75;       % Price stickiness
alpha = 0.33;       % Capital share
delta = 0.025;      % Depreciation rate

% Open economy parameters
phi_x = 0.3;        % Export sensitivity
phi_m = 0.3;        % Import sensitivity
gamma_fx = 0.2;     % Exchange rate sensitivity

% Behavioral parameters
myopia = 0.1;       % Myopia bias
overconfidence = 0.1; % Overconfidence bias

%% Initialize Variables
T = 100;
output = zeros(T,1);
exchange_rate = zeros(T,1);
exports = zeros(T,1);
imports = zeros(T,1);
net_exports = zeros(T,1);

%% Initial Conditions
output(1) = 1;
exchange_rate(1) = 1;
exports(1) = 0.3;
imports(1) = 0.2;

%% Simulation
for t = 2:T
    % Exchange rate shock
    fx_shock = gamma_fx * randn;
    exchange_rate(t) = exchange_rate(t-1) * (1 + fx_shock);
    
    % Exports and Imports with Behavioral Bias
    exports(t) = exports(t-1) + phi_x * exchange_rate(t-1) * (1 - myopia);
    imports(t) = imports(t-1) + phi_m * exchange_rate(t-1) * (1 + overconfidence);
    
    % Net exports
    net_exports(t) = exports(t) - imports(t);
    
    % Output dynamics
    output(t) = output(t-1) + net_exports(t) - delta * output(t-1);
end

%% Plot Results
figure;
subplot(3,1,1);
plot(output, 'LineWidth', 1.5);
title('Output Dynamics with Behavioral Biases in Open Economy');
xlabel('Time'); ylabel('Output'); grid on;

subplot(3,1,2);
plot(exchange_rate, 'LineWidth', 1.5);
title('Exchange Rate Dynamics');
xlabel('Time'); ylabel('Exchange Rate'); grid on;

subplot(3,1,3);
plot(net_exports, 'LineWidth', 1.5);
title('Net Exports Dynamics');
xlabel('Time'); ylabel('Net Exports'); grid on;
%% Open Economy DSGE with Behavioral Biases and Pricing Paradigms
clear; clc; close all;

%% Parameters
beta = 0.99;        % Discount factor
sigma = 1;          % Risk aversion
phi = 1.5;          % Taylor rule inflation response
theta = 0.75;       % Price stickiness
alpha = 0.33;       % Capital share
delta = 0.025;      % Depreciation rate

% Open economy parameters
phi_x = 0.3;        % Export sensitivity
phi_m = 0.3;        % Import sensitivity
gamma_fx = 0.2;     % Exchange rate sensitivity

% Behavioral parameters
myopia = 0.1;       % Myopia bias
overconfidence = 0.1; % Overconfidence bias

% Pricing paradigms parameters
pricing_mode = 'DCP';  % Choose between 'DCP', 'PCP', 'LCP'
markup_variation = 0.05; % Price adjustment sensitivity to exchange rates

%% Initialize Variables
T = 100;
output = zeros(T,1);
exchange_rate = zeros(T,1);
exports = zeros(T,1);
imports = zeros(T,1);
net_exports = zeros(T,1);
prices_domestic = zeros(T,1);
prices_foreign = zeros(T,1);

%% Initial Conditions
output(1) = 1;
exchange_rate(1) = 1;
exports(1) = 0.3;
imports(1) = 0.2;
prices_domestic(1) = 1;
prices_foreign(1) = 1;

%% Simulation
for t = 2:T
    % Exchange rate shock
    fx_shock = gamma_fx * randn;
    exchange_rate(t) = exchange_rate(t-1) * (1 + fx_shock);
    
    % Pricing Paradigm Effect
    switch pricing_mode
        case 'DCP'  % Dominant Currency Pricing
            prices_foreign(t) = prices_foreign(t-1);
        case 'PCP'  % Producer Currency Pricing
            prices_foreign(t) = prices_domestic(t-1) * (1 + markup_variation * fx_shock);
        case 'LCP'  % Local Currency Pricing
            prices_foreign(t) = prices_foreign(t-1) * (1 + markup_variation * fx_shock);
    end
    
    % Exports and Imports with Behavioral Bias
    exports(t) = exports(t-1) + phi_x * exchange_rate(t-1) * (1 - myopia) * prices_foreign(t);
    imports(t) = imports(t-1) + phi_m * exchange_rate(t-1) * (1 + overconfidence);
    
    % Net exports
    net_exports(t) = exports(t) - imports(t);
    
    % Output dynamics
    output(t) = output(t-1) + net_exports(t) - delta * output(t-1);
end

%% Plot Results
figure;
subplot(4,1,1);
plot(output, 'LineWidth', 1.5);
title('Output Dynamics with Behavioral Biases and Pricing Paradigms');
xlabel('Time'); ylabel('Output'); grid on;

subplot(4,1,2);
plot(exchange_rate, 'LineWidth', 1.5);
title('Exchange Rate Dynamics');
xlabel('Time'); ylabel('Exchange Rate'); grid on;

subplot(4,1,3);
plot(net_exports, 'LineWidth', 1.5);
title('Net Exports Dynamics');
xlabel('Time'); ylabel('Net Exports'); grid on;

subplot(4,1,4);
plot(prices_foreign, 'LineWidth', 1.5);
title('Foreign Prices Dynamics (Pricing Paradigm Impact)');
xlabel('Time'); ylabel('Prices'); grid on;
%% Open Economy DSGE with Behavioral Biases and Pricing Paradigms
clear; clc; close all;

%% Parameters
beta = 0.99;        % Discount factor
sigma = 1;          % Risk aversion
phi_pi = 1.5;       % Taylor rule inflation response (from paper)
phi_y = 0.5;        % Taylor rule output gap response (from paper)
theta = 0.75;       % Price stickiness
alpha = 0.33;       % Capital share
delta = 0.025;      % Depreciation rate

% Open economy parameters
phi_x = 0.3;        % Export sensitivity
phi_m = 0.3;        % Import sensitivity
gamma_fx = 0.2;     % Exchange rate sensitivity

% Behavioral parameters
myopia = 0.1;       % Myopia bias
overconfidence = 0.1; % Overconfidence bias

% Pricing paradigms parameters
pricing_mode = 'DCP';  % Choose between 'DCP', 'PCP', 'LCP'
markup_variation = 0.05; % Price adjustment sensitivity to exchange rates

% Financial frictions parameters
spread_sensitivity = 0.1; % Sensitivity of interest rate spread to output gap
credit_constraint = 0.2;  % Credit constraints affecting consumption/investment

%% Initialize Variables
T = 100;
output = zeros(T,1);
exchange_rate = zeros(T,1);
exports = zeros(T,1);
imports = zeros(T,1);
net_exports = zeros(T,1);
prices_domestic = zeros(T,1);
prices_foreign = zeros(T,1);
interest_rate = zeros(T,1);
spread = zeros(T,1);

%% Initial Conditions
output(1) = 1;
exchange_rate(1) = 1;
exports(1) = 0.3;
imports(1) = 0.2;
prices_domestic(1) = 1;
prices_foreign(1) = 1;
interest_rate(1) = 0.05;
spread(1) = 0.01;

%% Simulation
for t = 2:T
    % Exchange rate shock
    fx_shock = gamma_fx * randn;
    exchange_rate(t) = exchange_rate(t-1) * (1 + fx_shock);
    
    % Pricing Paradigm Effect
    switch pricing_mode
        case 'DCP'  % Dominant Currency Pricing
            prices_foreign(t) = prices_foreign(t-1);
        case 'PCP'  % Producer Currency Pricing
            prices_foreign(t) = prices_domestic(t-1) * (1 + markup_variation * fx_shock);
        case 'LCP'  % Local Currency Pricing
            prices_foreign(t) = prices_foreign(t-1) * (1 + markup_variation * fx_shock);
    end
    
    % Interest Rate with Taylor Rule
    interest_rate(t) = interest_rate(t-1) + phi_pi * (prices_domestic(t-1) - 1) + phi_y * (output(t-1) - 1);
    
    % Financial Spread Adjustment
    spread(t) = spread_sensitivity * (output(t-1) - 1);
    
    % Exports and Imports with Behavioral Bias and Credit Constraints
    exports(t) = exports(t-1) + phi_x * exchange_rate(t-1) * (1 - myopia) * prices_foreign(t);
    imports(t) = imports(t-1) + phi_m * exchange_rate(t-1) * (1 + overconfidence + credit_constraint * spread(t));
    
    % Net exports
    net_exports(t) = exports(t) - imports(t);
    
    % Output dynamics
    output(t) = output(t-1) + net_exports(t) - delta * output(t-1);
end

%% Plot Results
figure;
subplot(5,1,1);
plot(output, 'LineWidth', 1.5);
title('Output Dynamics with Financial Frictions and Pricing Paradigms');
xlabel('Time'); ylabel('Output'); grid on;

subplot(5,1,2);
plot(exchange_rate, 'LineWidth', 1.5);
title('Exchange Rate Dynamics');
xlabel('Time'); ylabel('Exchange Rate'); grid on;

subplot(5,1,3);
plot(net_exports, 'LineWidth', 1.5);
title('Net Exports Dynamics');
xlabel('Time'); ylabel('Net Exports'); grid on;

subplot(5,1,4);
plot(interest_rate, 'LineWidth', 1.5);
title('Interest Rate Dynamics (Taylor Rule)');
xlabel('Time'); ylabel('Interest Rate'); grid on;

subplot(5,1,5);
plot(spread, 'LineWidth', 1.5);
title('Financial Spread Dynamics');
xlabel('Time'); ylabel('Spread'); grid on;
