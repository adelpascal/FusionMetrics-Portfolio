%% Heterogeneous Agent New Keynesian (HANK) Model
% Author: Adelia Nur Fasha
% Objective: Simulate a HANK model with Savers and Borrowers

clc; clear; close all;

%% Model Parameters
beta_s = 0.98;      % Discount factor for Savers
beta_b = 0.96;      % Discount factor for Borrowers
sigma = 1;          % Risk aversion (CRRA)
phi = 1.5;          % Phillips curve parameter
theta = 0.75;       % Calvo price stickiness
rho_income = 0.9;   % Persistence of income shock
sigma_income = 0.02; % Std. dev. of income shock
r_bar = 0.03;       % Steady-state interest rate

T = 100;            % Time horizon
shock_size = 0.01;  % Size of monetary policy shock

%% Initialize Variables
income_savers = ones(T,1);
income_borrowers = ones(T,1);
consumption_savers = zeros(T,1);
consumption_borrowers = zeros(T,1);
inflation = zeros(T,1);
interest_rate = r_bar * ones(T,1);
output_gap = zeros(T,1);

%% Generate Income Shocks
rng('default');
income_shock_savers = sigma_income * randn(T,1);
income_shock_borrowers = sigma_income * randn(T,1);

for t = 2:T
    income_savers(t) = rho_income * income_savers(t-1) + income_shock_savers(t);
    income_borrowers(t) = rho_income * income_borrowers(t-1) + income_shock_borrowers(t);
end
%% Consumption Decisions
for t = 2:T
    % Savers' consumption (depends on income and interest rate)
    consumption_savers(t) = (beta_s * (1 + interest_rate(t-1)))^(-1/sigma) * income_savers(t);

    % Borrowers face borrowing constraints, more sensitive to income shocks
    consumption_borrowers(t) = max(0, income_borrowers(t) - 0.5 * shock_size);
end
%% Aggregate Demand and Monetary Policy Rule
for t = 2:T
    % Aggregate consumption (weighted average)
    total_consumption = 0.7 * consumption_savers(t) + 0.3 * consumption_borrowers(t);
    
    % Output gap (difference from steady state)
    output_gap(t) = total_consumption - 1;

    % Taylor rule for interest rate (monetary policy response)
    interest_rate(t) = r_bar + phi * inflation(t-1) + 0.5 * output_gap(t);
    
    % Inflation dynamics (Phillips curve)
    inflation(t) = inflation(t-1) + theta * output_gap(t);
end
%% Visualization
figure;
subplot(3,1,1);
plot(1:T, consumption_savers, 'b-', 'LineWidth', 2); hold on;
plot(1:T, consumption_borrowers, 'r--', 'LineWidth', 2);
xlabel('Time'); ylabel('Consumption');
legend('Savers', 'Borrowers');
title('Consumption Dynamics');
grid on;

subplot(3,1,2);
plot(1:T, inflation, 'k-', 'LineWidth', 2);
xlabel('Time'); ylabel('Inflation');
title('Inflation Dynamics');
grid on;

subplot(3,1,3);
plot(1:T, interest_rate, 'm-', 'LineWidth', 2);
xlabel('Time'); ylabel('Interest Rate');
title('Monetary Policy Response');
grid on;
% Define Parameters
credit_limit = -0.5;  % Borrowers cannot borrow more than this (negative means debt)

% Initialize Consumption for Savers and Borrowers
consumption_savers = zeros(T, 1);
consumption_borrowers = zeros(T, 1);
wealth_borrowers = zeros(T, 1); % Track borrowers' wealth

% Loop through Time Periods
for t = 2:T
    % Income Shocks (for demonstration, random normal shocks)
    income_shock = sigma_income * randn;

    % Update Borrowers' Wealth with Income Shock
    wealth_borrowers(t) = wealth_borrowers(t-1) + income_shock;

    % Apply Borrowing Constraint
    if wealth_borrowers(t) < credit_limit
        wealth_borrowers(t) = credit_limit;
    end

    % Consumption Dynamics
    consumption_savers(t) = rho_income * consumption_savers(t-1) + shock_size * randn;
    consumption_borrowers(t) = rho_income * wealth_borrowers(t) + shock_size * randn;
end

% Plot Results
figure;
plot(consumption_savers, 'b', 'LineWidth', 1.5); hold on;
plot(consumption_borrowers, 'r--', 'LineWidth', 1.5);
legend('Savers', 'Borrowers');
title('Consumption Dynamics with Borrowing Constraints');
xlabel('Time');
ylabel('Consumption');
grid on;
% Define Parameters for Income Shocks
shock_probability = 0.1;     % 10% chance of a negative shock
shock_magnitude = -0.3;      % Size of the negative shock

% Initialize Consumption for Savers and Borrowers
consumption_savers = zeros(T, 1);
consumption_borrowers = zeros(T, 1);
wealth_borrowers = zeros(T, 1);

% Loop through Time Periods
for t = 2:T
    % Random Income Shock Occurrence
    if rand < shock_probability
        income_shock_savers = shock_magnitude * rand;
        income_shock_borrowers = 2 * shock_magnitude * rand; % Larger shocks for borrowers
    else
        income_shock_savers = sigma_income * randn;
        income_shock_borrowers = sigma_income * randn;
    end

    % Update Borrowers' Wealth with Income Shock
    wealth_borrowers(t) = wealth_borrowers(t-1) + income_shock_borrowers;

    % Apply Borrowing Constraint
    if wealth_borrowers(t) < credit_limit
        wealth_borrowers(t) = credit_limit;
    end

    % Consumption Dynamics with Income Shocks
    consumption_savers(t) = rho_income * consumption_savers(t-1) + income_shock_savers;
    consumption_borrowers(t) = rho_income * wealth_borrowers(t) + income_shock_borrowers;
end

% Plot Results
figure;
plot(consumption_savers, 'b', 'LineWidth', 1.5); hold on;
plot(consumption_borrowers, 'r--', 'LineWidth', 1.5);
legend('Savers', 'Borrowers');
title('Consumption Dynamics with Borrowing Constraints and Income Shocks');
xlabel('Time');
ylabel('Consumption');
grid on;
% Taylor Rule Parameters
phi_pi = 1.5;   % Response to inflation
phi_y = 0.5;    % Response to output gap
inflation_target = 0.02;  % 2% Inflation target
output_gap = zeros(T,1);  % Initialize output gap
inflation = zeros(T,1);   % Initialize inflation
interest_rate = zeros(T,1);  % Initialize interest rate

% Simulate Monetary Policy Response
for t = 2:T
    % Calculate Output Gap
    output_gap(t) = (consumption_savers(t) + consumption_borrowers(t)) - ...
                    (consumption_savers(t-1) + consumption_borrowers(t-1));

    % Simulate Inflation Dynamics (Simplified)
    inflation(t) = inflation(t-1) + 0.1 * output_gap(t);

    % Apply Taylor Rule for Interest Rate
    interest_rate(t) = phi_pi * (inflation(t) - inflation_target) + phi_y * output_gap(t);

    % Update Borrowers' Wealth with Interest Payments
    wealth_borrowers(t) = wealth_borrowers(t) - interest_rate(t) * wealth_borrowers(t-1);

    % Apply Borrowing Constraint Again
    if wealth_borrowers(t) < credit_limit
        wealth_borrowers(t) = credit_limit;
    end

    % Update Consumption with Interest Rate Effect
    consumption_savers(t) = consumption_savers(t) - 0.1 * interest_rate(t);
    consumption_borrowers(t) = consumption_borrowers(t) - 0.3 * interest_rate(t);
end

% Plot Results: Interest Rate, Inflation, and Consumption
figure;

subplot(3,1,1);
plot(consumption_savers, 'b', 'LineWidth', 1.5); hold on;
plot(consumption_borrowers, 'r--', 'LineWidth', 1.5);
legend('Savers', 'Borrowers');
title('Consumption with Monetary Policy');
ylabel('Consumption');
grid on;

subplot(3,1,2);
plot(inflation, 'k', 'LineWidth', 1.5);
title('Inflation Dynamics');
ylabel('Inflation');
grid on;

subplot(3,1,3);
plot(interest_rate, 'm', 'LineWidth', 1.5);
title('Interest Rate Response (Taylor Rule)');
xlabel('Time');
ylabel('Interest Rate');
grid on;
