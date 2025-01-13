% Operational Risk Simulation in Supply Chain with Compliance Focus
% Author: Adelia Nur Fasha
% Objective: Simulate operational risks in supply chain processes with a focus on compliance and internal control implementation.

clc; clear; close all;

%% Simulation Parameters
numSimulations = 10000;   % Number of Monte Carlo simulations
numProcesses = 5;         % Number of supply chain processes

% Process Risk Probabilities (example probabilities for each stage, including compliance risk)
riskProbabilities = [0.05, 0.1, 0.08, 0.06, 0.12];
complianceRiskFactor = 0.03; % Additional compliance risk factor

% Impact of each process failure (in monetary loss, including compliance penalties)
processImpact = [50000, 100000, 75000, 60000, 120000];
complianceImpact = 80000; % Impact of compliance violations

totalLosses = zeros(1, numSimulations);
complianceViolations = zeros(1, numSimulations);

%% Monte Carlo Simulation
for i = 1:numSimulations
    failureEvents = rand(1, numProcesses) < riskProbabilities; % Simulate process failures
    complianceViolation = rand < complianceRiskFactor;         % Simulate compliance failure
    totalLosses(i) = sum(failureEvents .* processImpact) + (complianceViolation * complianceImpact);
    complianceViolations(i) = complianceViolation;
end

%% Analysis
expectedLoss = mean(totalLosses);
maxLoss = max(totalLosses);
probZeroLoss = sum(totalLosses == 0) / numSimulations;
complianceViolationRate = sum(complianceViolations) / numSimulations;

%% Visualization
figure;
histogram(totalLosses, 'BinWidth', 10000, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Total Loss per Simulation');
ylabel('Frequency');
title('Operational & Compliance Risk Simulation - Supply Chain');
grid on;

%% Results
fprintf('Expected Loss: $%.2f\n', expectedLoss);
fprintf('Maximum Potential Loss: $%.2f\n', maxLoss);
fprintf('Probability of Zero Loss: %.2f%%\n', probZeroLoss * 100);
fprintf('Compliance Violation Rate: %.2f%%\n', complianceViolationRate * 100);

%% Risk Mitigation Suggestion
if expectedLoss > 50000
    fprintf('Recommendation: Strengthen internal controls and compliance monitoring.\n');
else
    fprintf('Current risk levels are acceptable.\n');
end

if complianceViolationRate > 0.05
    fprintf('Recommendation: Implement additional compliance training and regular audits.\n');
end
% Advanced Operational Risk Simulation with Compliance Focus and Detailed Analysis
% Author: Adelia Nur Fasha
% Objective: Simulate operational risks in supply chain processes with a focus on compliance and internal control implementation, enhanced with advanced risk analysis and visualization.

clc; clear; close all;

%% Simulation Parameters
numSimulations = 10000;   % Number of Monte Carlo simulations
numProcesses = 5;         % Number of supply chain processes

% Process Risk Probabilities (example probabilities for each stage, including compliance risk)
riskProbabilities = [0.05, 0.1, 0.08, 0.06, 0.12];
complianceRiskFactor = 0.03; % Additional compliance risk factor

% Impact of each process failure (in monetary loss, including compliance penalties)
processImpact = [50000, 100000, 75000, 60000, 120000];
complianceImpact = 80000; % Impact of compliance violations

totalLosses = zeros(1, numSimulations);
complianceViolations = zeros(1, numSimulations);
processFailures = zeros(numSimulations, numProcesses);

%% Monte Carlo Simulation
for i = 1:numSimulations
    failureEvents = rand(1, numProcesses) < riskProbabilities; % Simulate process failures
    complianceViolation = rand < complianceRiskFactor;         % Simulate compliance failure
    totalLosses(i) = sum(failureEvents .* processImpact) + (complianceViolation * complianceImpact);
    complianceViolations(i) = complianceViolation;
    processFailures(i, :) = failureEvents; % Track process-specific failures
end

%% Analysis
expectedLoss = mean(totalLosses);
maxLoss = max(totalLosses);
probZeroLoss = sum(totalLosses == 0) / numSimulations;
complianceViolationRate = sum(complianceViolations) / numSimulations;
avgFailuresPerProcess = mean(processFailures);

%% Root Cause Analysis
[maxFailureRate, riskiestProcess] = max(avgFailuresPerProcess);

%% Visualization - Loss Distribution
figure;
histogram(totalLosses, 'BinWidth', 10000, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Total Loss per Simulation');
ylabel('Frequency');
title('Operational & Compliance Risk Simulation - Supply Chain');
grid on;

%% Visualization - Process Risk Contribution
figure;
bar(avgFailuresPerProcess, 'FaceColor', [0.8 0.3 0.3]);
xlabel('Supply Chain Processes');
ylabel('Average Failure Rate');
title('Process Risk Contribution Analysis');
grid on;

%% Visualization - Compliance vs Operational Risk
figure;
pie([sum(complianceViolations), numSimulations - sum(complianceViolations)], ...
    [1 0], {'Compliance Violations', 'No Violations'});
title('Compliance Risk Occurrence');

%% Results
fprintf('Expected Loss: $%.2f\n', expectedLoss);
fprintf('Maximum Potential Loss: $%.2f\n', maxLoss);
fprintf('Probability of Zero Loss: %.2f%%\n', probZeroLoss * 100);
fprintf('Compliance Violation Rate: %.2f%%\n', complianceViolationRate * 100);
fprintf('Process %d has the highest failure rate of %.2f%%\n', riskiestProcess, maxFailureRate * 100);

%% Risk Mitigation Suggestion
if expectedLoss > 50000
    fprintf('Recommendation: Strengthen internal controls and compliance monitoring.\n');
else
    fprintf('Current risk levels are acceptable.\n');
end

if complianceViolationRate > 0.05
    fprintf('Recommendation: Implement additional compliance training and regular audits.\n');
end

if maxFailureRate > 0.1
    fprintf('Recommendation: Focus on improving process %d to reduce operational risks.\n', riskiestProcess);
end
% Comprehensive Industry-Wide Risk Management Simulation
% Author: Adelia Nur Fasha
% Objective: Simulate and analyze key risk types across industries (Banking, Finance, Manufacturing) following industry standards (Basel III, ISO 31000).

clc; clear; close all;

%% Simulation Parameters
numSimulations = 10000;   % Number of Monte Carlo simulations

%% 1. Credit Risk (Banking & Finance)
% Probability of default for different borrower categories
creditRiskProbabilities = [0.02, 0.05, 0.1]; % Low, Medium, High risk borrowers
creditExposure = [500000, 300000, 200000];  % Exposure amounts

%% 2. Market Risk (Finance)
% Daily volatility of asset classes (stocks, bonds, commodities)
marketVolatility = [0.015, 0.01, 0.02];  % Stocks, Bonds, Commodities
marketExposure = [1e6, 5e5, 2e5];       % Investment amounts

%% 3. Operational Risk (Manufacturing)
% Process failure probabilities and impacts
operationalRiskProbabilities = [0.03, 0.06, 0.02, 0.08, 0.1]; % Process failure risk
operationalImpact = [80000, 150000, 50000, 120000, 200000];   % Financial impact

%% 4. Compliance Risk (Cross-Industry)
% Regulatory compliance breach probability and penalties
complianceRiskFactor = 0.025;
compliancePenalty = 100000;

%% Simulation Variables
creditLosses = zeros(1, numSimulations);
marketLosses = zeros(1, numSimulations);
operationalLosses = zeros(1, numSimulations);
complianceLosses = zeros(1, numSimulations);
totalLosses = zeros(1, numSimulations);

%% Monte Carlo Simulation
for i = 1:numSimulations
    %% Credit Risk Simulation
    creditDefaults = rand(1, length(creditRiskProbabilities)) < creditRiskProbabilities;
    creditLosses(i) = sum(creditDefaults .* creditExposure);
    
    %% Market Risk Simulation (Value-at-Risk)
    marketShocks = randn(1, length(marketVolatility)) .* marketVolatility;
    marketLosses(i) = sum(marketShocks .* marketExposure);
    
    %% Operational Risk Simulation
    operationalFailures = rand(1, length(operationalRiskProbabilities)) < operationalRiskProbabilities;
    operationalLosses(i) = sum(operationalFailures .* operationalImpact);
    
    %% Compliance Risk Simulation
    complianceViolation = rand < complianceRiskFactor;
    complianceLosses(i) = complianceViolation * compliancePenalty;
    
    %% Total Risk Aggregation
    totalLosses(i) = creditLosses(i) + marketLosses(i) + operationalLosses(i) + complianceLosses(i);
end

%% Advanced Analysis
expectedTotalLoss = mean(totalLosses);
maxTotalLoss = max(totalLosses);
VaR95 = prctile(totalLosses, 5); % 95% Value-at-Risk (VaR)

%% Visualization - Loss Distribution
figure;
histogram(totalLosses, 'BinWidth', 50000, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Total Loss per Simulation');
ylabel('Frequency');
title('Comprehensive Risk Simulation - Industry-Wide');
grid on;

%% Visualization - Risk Breakdown
figure;
bar([mean(creditLosses), mean(marketLosses), mean(operationalLosses), mean(complianceLosses)], 'FaceColor', [0.8 0.3 0.3]);
xlabel('Risk Types');
ylabel('Average Loss');
title('Average Loss by Risk Type');
set(gca, 'XTickLabel', {'Credit', 'Market', 'Operational', 'Compliance'});
grid on;

%% Results
fprintf('Expected Total Loss: $%.2f\n', expectedTotalLoss);
fprintf('Maximum Total Loss: $%.2f\n', maxTotalLoss);
fprintf('95%% Value-at-Risk (VaR): $%.2f\n', VaR95);

%% Risk Mitigation Strategies
fprintf('\n--- Risk Mitigation Recommendations ---\n');
if mean(creditLosses) > 200000
    fprintf('Credit Risk: Diversify loan portfolios and strengthen credit assessments.\n');
end
if mean(marketLosses) > 100000
    fprintf('Market Risk: Implement hedging strategies and stress testing.\n');
end
if mean(operationalLosses) > 100000
    fprintf('Operational Risk: Enhance internal controls and automate critical processes.\n');
end
if mean(complianceLosses) > 50000
    fprintf('Compliance Risk: Increase regulatory audits and compliance training.\n');
end
fprintf('--------------------------------------\n');
% Risk_Simulation.m
% Advanced Risk Management Simulation in MATLAB
% Author: Adelia Nur Fasha

clc; clear; close all;

%% Simulation Settings
numSimulations = 10000;  % Number of Monte Carlo simulations

%% 1. Credit Risk Parameters
EAD = 1e6;               % Exposure at Default ($1 million)
PD = 0.02;               % Probability of Default (2%)
LGD = 0.45;              % Loss Given Default (45%)

%% 2. Market Risk Parameters
marketVolatility = 0.015;  % 1.5% daily volatility
marketExposure = 2e6;      % $2 million investment

%% 3. Operational Risk Parameters
opIncidentRate = 0.03;     % 3% chance of failure
opImpact = 5e5;            % $500,000 impact

%% 4. Compliance Risk Parameters
complianceViolationRate = 0.01;  % 1% violation probability
compliancePenalty = 1e5;         % $100,000 penalty

fprintf('✅ Risk Parameters Initialized in MATLAB!\n');
% Advanced Risk Management Simulation in MATLAB - Step 2
% Author: Adelia Nur Fasha

clc; clear; close all;

%% Simulation Settings
numSimulations = 10000;  % Number of Monte Carlo simulations

%% 1. Credit Risk Parameters
EAD = 1e6;               % Exposure at Default ($1 million)
PD = 0.02;               % Probability of Default (2%)
LGD = 0.45;              % Loss Given Default (45%)

%% 2. Market Risk Parameters
marketVolatility = 0.015;  % 1.5% daily volatility
marketExposure = 2e6;      % $2 million investment

%% 3. Operational Risk Parameters
opIncidentRate = 0.03;     % 3% chance of operational failure
opImpact = 5e5;            % $500,000 impact

%% 4. Compliance Risk Parameters
complianceViolationRate = 0.01;  % 1% violation probability
compliancePenalty = 1e5;         % $100,000 penalty

%% Initialize Arrays to Store Results
creditLosses = zeros(1, numSimulations);
marketLosses = zeros(1, numSimulations);
operationalLosses = zeros(1, numSimulations);
complianceLosses = zeros(1, numSimulations);
totalLosses = zeros(1, numSimulations);

%% Monte Carlo Simulation
for i = 1:numSimulations
    % Credit Risk Simulation
    creditDefault = rand < PD;
    creditLosses(i) = creditDefault * EAD * LGD;
    
    % Market Risk Simulation
    marketShock = randn * marketVolatility;
    marketLosses(i) = abs(marketShock * marketExposure);
    
    % Operational Risk Simulation
    operationalFailure = rand < opIncidentRate;
    operationalLosses(i) = operationalFailure * opImpact;
    
    % Compliance Risk Simulation
    complianceViolation = rand < complianceViolationRate;
    complianceLosses(i) = complianceViolation * compliancePenalty;
    
    % Total Loss Aggregation
    totalLosses(i) = creditLosses(i) + marketLosses(i) + operationalLosses(i) + complianceLosses(i);
end

%% Summary Statistics
expectedTotalLoss = mean(totalLosses);
maxTotalLoss = max(totalLosses);
VaR95 = prctile(totalLosses, 5);  % 95% Value-at-Risk

%% Display Results
fprintf('--- Risk Simulation Results ---\n');
fprintf('Expected Total Loss: $%.2f\n', expectedTotalLoss);
fprintf('Maximum Total Loss: $%.2f\n', maxTotalLoss);
fprintf('95%% Value-at-Risk (VaR): $%.2f\n', VaR95);

%% Visualization - Total Loss Distribution
figure;
histogram(totalLosses, 'BinWidth', 10000, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Total Loss ($)');
ylabel('Frequency');
title('Total Loss Distribution from Risk Simulation');
grid on;
%% Step 3: Stress Testing and Scenario Analysis

% Stress Factors for Extreme Events
stressFactor_CreditPD = 2.5;    % 2.5x increase in Probability of Default
stressFactor_MarketVol = 3;     % 3x increase in market volatility
stressFactor_OpRisk = 2;        % 2x increase in operational risk
stressFactor_Compliance = 1.5;  % 1.5x increase in compliance violation

% Initialize Stress Test Losses
stressTestLosses = zeros(1, numSimulations);

%% Stress Testing Simulation
for i = 1:numSimulations
    % Credit Risk under Stress
    stressedPD = PD * stressFactor_CreditPD;
    creditDefaultStress = rand < stressedPD;
    creditLossStress = creditDefaultStress * EAD * LGD;
    
    % Market Risk under Stress
    stressedMarketShock = randn * marketVolatility * stressFactor_MarketVol;
    marketLossStress = abs(stressedMarketShock * marketExposure);
    
    % Operational Risk under Stress
    stressedOpIncident = rand < (opIncidentRate * stressFactor_OpRisk);
    operationalLossStress = stressedOpIncident * opImpact;
    
    % Compliance Risk under Stress
    stressedComplianceViolation = rand < (complianceViolationRate * stressFactor_Compliance);
    complianceLossStress = stressedComplianceViolation * compliancePenalty;
    
    % Total Stressed Loss
    stressTestLosses(i) = creditLossStress + marketLossStress + operationalLossStress + complianceLossStress;
end

%% Stress Test Analysis
expectedStressLoss = mean(stressTestLosses);
maxStressLoss = max(stressTestLosses);
VaR95_Stress = prctile(stressTestLosses, 5);

%% Display Stress Test Results
fprintf('\n--- Stress Test Results ---\n');
fprintf('Expected Stress Loss: $%.2f\n', expectedStressLoss);
fprintf('Maximum Stress Loss: $%.2f\n', maxStressLoss);
fprintf('95%% VaR under Stress: $%.2f\n', VaR95_Stress);

%% Visualization - Stress Testing vs. Baseline
figure;
boxplot([totalLosses', stressTestLosses'], 'Labels', {'Baseline Scenario', 'Stress Test Scenario'});
ylabel('Total Loss ($)');
title('Comparison of Baseline vs. Stress Test Losses');
grid on;
%% Step 4: Correlation Analysis Between Risks

% Define a Correlation Matrix for the 4 risk types
% [Credit, Market, Operational, Compliance]
correlationMatrix = [1, 0.5, 0.3, 0.2;   % Credit Risk
                     0.5, 1, 0.4, 0.3;   % Market Risk
                     0.3, 0.4, 1, 0.5;   % Operational Risk
                     0.2, 0.3, 0.5, 1];  % Compliance Risk

% Generate correlated risk factors using multivariate normal distribution
rng('default'); % For reproducibility
correlatedRiskFactors = mvnrnd(zeros(1, 4), correlationMatrix, numSimulations);

% Initialize Correlated Losses
correlatedTotalLosses = zeros(1, numSimulations);

%% Monte Carlo Simulation with Correlated Risks
for i = 1:numSimulations
    % Credit Risk (Correlated)
    creditDefaultCorr = correlatedRiskFactors(i, 1) > norminv(1 - PD);
    creditLossCorr = creditDefaultCorr * EAD * LGD;
    
    % Market Risk (Correlated)
    marketShockCorr = correlatedRiskFactors(i, 2) * marketVolatility;
    marketLossCorr = abs(marketShockCorr * marketExposure);
    
    % Operational Risk (Correlated)
    opIncidentCorr = correlatedRiskFactors(i, 3) > norminv(1 - opIncidentRate);
    operationalLossCorr = opIncidentCorr * opImpact;
    
    % Compliance Risk (Correlated)
    complianceViolationCorr = correlatedRiskFactors(i, 4) > norminv(1 - complianceViolationRate);
    complianceLossCorr = complianceViolationCorr * compliancePenalty;
    
    % Total Correlated Loss Calculation
    correlatedTotalLosses(i) = creditLossCorr + marketLossCorr + operationalLossCorr + complianceLossCorr;
end

%% Analysis of Correlated Losses
expectedCorrLoss = mean(correlatedTotalLosses);
maxCorrLoss = max(correlatedTotalLosses);
VaR95_Corr = prctile(correlatedTotalLosses, 5);

%% Display Correlation Analysis Results
fprintf('\n--- Correlation Analysis Results ---\n');
fprintf('Expected Loss with Correlation: $%.2f\n', expectedCorrLoss);
fprintf('Maximum Loss with Correlation: $%.2f\n', maxCorrLoss);
fprintf('95%% VaR with Correlation: $%.2f\n', VaR95_Corr);

%% Visualization - Correlated vs. Independent Losses
figure;
boxplot([totalLosses', correlatedTotalLosses'], 'Labels', {'Independent Risks', 'Correlated Risks'});
ylabel('Total Loss ($)');
title('Impact of Risk Correlation on Total Losses');
grid on;
%% Step 5: Risk Mitigation Strategy Simulation

% Mitigation Effectiveness (Reduction Percentages)
mitigation_Credit = 0.30;   % 30% reduction in credit risk
mitigation_Market = 0.25;   % 25% reduction in market risk
mitigation_Operational = 0.40; % 40% reduction in operational risk
mitigation_Compliance = 0.50;  % 50% reduction in compliance risk

% Initialize Mitigated Losses
mitigatedTotalLosses = zeros(1, numSimulations);

%% Monte Carlo Simulation with Mitigation
for i = 1:numSimulations
    % Credit Risk with Mitigation
    creditDefaultMitigated = rand < PD;
    creditLossMitigated = creditDefaultMitigated * EAD * LGD * (1 - mitigation_Credit);
    
    % Market Risk with Mitigation
    marketShockMitigated = randn * marketVolatility;
    marketLossMitigated = abs(marketShockMitigated * marketExposure * (1 - mitigation_Market));
    
    % Operational Risk with Mitigation
    operationalFailureMitigated = rand < opIncidentRate;
    operationalLossMitigated = operationalFailureMitigated * opImpact * (1 - mitigation_Operational);
    
    % Compliance Risk with Mitigation
    complianceViolationMitigated = rand < complianceViolationRate;
    complianceLossMitigated = complianceViolationMitigated * compliancePenalty * (1 - mitigation_Compliance);
    
    % Total Mitigated Loss Calculation
    mitigatedTotalLosses(i) = creditLossMitigated + marketLossMitigated + operationalLossMitigated + complianceLossMitigated;
end

%% Analysis of Mitigated Losses
expectedMitigatedLoss = mean(mitigatedTotalLosses);
maxMitigatedLoss = max(mitigatedTotalLosses);
VaR95_Mitigated = prctile(mitigatedTotalLosses, 5);

%% Display Mitigation Results
fprintf('\n--- Risk Mitigation Simulation Results ---\n');
fprintf('Expected Loss after Mitigation: $%.2f\n', expectedMitigatedLoss);
fprintf('Maximum Loss after Mitigation: $%.2f\n', maxMitigatedLoss);
fprintf('95%% VaR after Mitigation: $%.2f\n', VaR95_Mitigated);

%% Visualization - Loss Comparison Before vs. After Mitigation
figure;
boxplot([totalLosses', mitigatedTotalLosses'], 'Labels', {'Before Mitigation', 'After Mitigation'});
ylabel('Total Loss ($)');
title('Impact of Mitigation Strategies on Total Losses');
grid on;
%% Advanced Credit Risk Simulation with Potential Future Exposure (PFE)
clc; clear; close all;

%% Simulation Parameters
numSimulations = 10000;      % Number of Monte Carlo simulations
exposureMaturity = 1:10;     % 10-year exposure horizon
confidenceLevel = 0.95;      % 95% confidence level for PFE

%% Counterparty Credit Parameters
EAD = 1e6;                   % Exposure at Default ($1,000,000)
LGD = 0.45;                  % Loss Given Default (45%)
PD = 0.02;                   % Probability of Default (2%)

%% Market Volatility Impact on Exposure
volatilityFactor = 0.15;     % Exposure volatility (15%)
driftRate = 0.03;            % Drift rate for exposure growth (3%)

%% Initialize Matrices
exposurePath = zeros(numSimulations, length(exposureMaturity));
pfe = zeros(1, length(exposureMaturity));

%% Monte Carlo Simulation for Exposure Paths
rng('default'); % For reproducibility
for i = 1:numSimulations
    for t = 1:length(exposureMaturity)
        % Simulate exposure over time with geometric Brownian motion
        randomShock = randn;
        exposurePath(i, t) = EAD * exp((driftRate - 0.5 * volatilityFactor^2) * t + volatilityFactor * sqrt(t) * randomShock);
    end
end

%% Calculate Potential Future Exposure (PFE)
for t = 1:length(exposureMaturity)
    pfe(t) = prctile(exposurePath(:, t), confidenceLevel * 100);
end

%% Credit Valuation Adjustment (CVA) Calculation
discountFactor = exp(-0.03 * exposureMaturity); % Assuming 3% risk-free rate
cva = LGD * PD * sum(pfe .* discountFactor);

%% Visualization - Exposure Path and PFE
figure;
plot(exposureMaturity, exposurePath(1:50, :), 'Color', [0.8, 0.8, 0.8]); hold on;
plot(exposureMaturity, pfe, 'r-', 'LineWidth', 2);
xlabel('Time Horizon (Years)');
ylabel('Exposure ($)');
title('Exposure Evolution and Potential Future Exposure (PFE)');
legend('Simulated Exposure Paths', '95% PFE');
grid on;

%% Display Results
fprintf('--- Advanced Credit Risk Simulation Results ---\n');
fprintf('Expected Exposure at Maturity: $%.2f\n', mean(exposurePath(:, end)));
fprintf('95%% Potential Future Exposure (PFE) at Year 10: $%.2f\n', pfe(end));
fprintf('Credit Valuation Adjustment (CVA): $%.2f\n', cva);
%% Basel III Stress Testing Enhancements
clc; clear; close all;

%% Define Simulation Parameters
numSimulations = 10000;  % Number of Monte Carlo simulations

%% Stress Testing Parameters
marketShockFactors = [1.2, 1.5, 2.0];  % Mild, Moderate, Severe market shocks
liquidityStressFactor = [1.1, 1.3, 1.6]; % Impact of liquidity crisis
creditSpreadShock = [0.02, 0.05, 0.1];  % Credit spread widening

%% Initialize Stress Test Results
stressTestLosses = zeros(numSimulations, length(marketShockFactors));

%% Define Missing Variables

EAD = 1e6;               % Exposure at Default (e.g., $1,000,000)
driftRate = 0.03;        % Expected growth rate (3%)
volatilityFactor = 0.2;  % Volatility factor (20%)
LGD = 0.45;              % Loss Given Default (45%)

%% Stress Testing Simulation
for j = 1:length(marketShockFactors)
    for i = 1:numSimulations
        % Apply market shock to exposure
        shockedExposure = EAD * marketShockFactors(j) * exp((driftRate - 0.5 * volatilityFactor^2) * 10 + volatilityFactor * sqrt(10) * randn);
        
        % Apply liquidity stress impact
        liquidityImpact = shockedExposure * liquidityStressFactor(j);
        
        % Apply credit spread widening
        creditSpreadImpact = liquidityImpact * (1 + creditSpreadShock(j));
        
        % Total stress scenario loss
        stressTestLosses(i, j) = LGD * creditSpreadImpact;
    end
end

%% Stress Testing Analysis
expectedStressLoss = mean(stressTestLosses);
maxStressLoss = max(stressTestLosses);
VaR95_Stress = prctile(stressTestLosses, 5);

%% Visualization - Stress Testing Losses
figure;
boxplot(stressTestLosses, 'Labels', {'Mild', 'Moderate', 'Severe'});
ylabel('Total Loss ($)');
title('Basel III Stress Testing Scenarios');
grid on;

%% Display Results
fprintf('--- Basel III Stress Testing Results ---\n');
for j = 1:length(marketShockFactors)
    fprintf('Scenario %d:\n', j);
    fprintf('Expected Stress Loss: $%.2f\n', expectedStressLoss(j));
    fprintf('Maximum Stress Loss: $%.2f\n', maxStressLoss(j));
    fprintf('95%% VaR under Stress: $%.2f\n\n', VaR95_Stress(j));
end
%% Visualization - Stress Testing Results
figure;
boxplot(stressTestLosses, 'Labels', {'Mild', 'Moderate', 'Severe'}, 'Colors', 'r');
xlabel('Stress Test Scenarios');
ylabel('Total Loss ($)');
title('Stress Testing Impact on Total Loss');
grid on;
% Simulating valid data for both risk factors
numSimulations = 10000;

% Random data for credit spread impact (avoid NaN)
creditSpreadImpact = rand(numSimulations, 1) * 1e6;  % Simulated monetary impact

% Random data for liquidity impact (avoid NaN)
liquidityImpact = rand(numSimulations, 1) * 5e5;     % Simulated liquidity impact
% Combine data into a matrix
riskDataMatrix = [creditSpreadImpact, liquidityImpact];

% Recalculate correlation
correlationMatrix = corr(riskDataMatrix, 'Rows', 'complete');  % Ignores NaN

% Plot corrected correlation matrix
figure;
heatmap(correlationMatrix, 'Colormap', jet, 'ColorbarVisible', 'on');
xlabel('Risk Factors');
ylabel('Risk Factors');
title('Corrected Correlation Matrix of Risk Factors');
%% Risk Decomposition
avgCreditLoss = mean(creditSpreadImpact);
avgLiquidityLoss = mean(liquidityImpact);

figure;
bar([avgCreditLoss, avgLiquidityLoss], 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTickLabel', {'Credit Risk', 'Liquidity Risk'});
ylabel('Average Loss ($)');
title('Risk Decomposition Analysis');
grid on;
%% Risk Mitigation Recommendations
fprintf('\n--- Risk Mitigation Recommendations ---\n');
if avgCreditLoss > 1e6
    fprintf('Credit Risk: Strengthen underwriting and diversify loan portfolios.\n');
end
if avgLiquidityLoss > 1e6
    fprintf('Liquidity Risk: Increase liquidity buffers and monitor funding gaps.\n');
end
fprintf('--------------------------------------\n');
%%DSGE Elaborate
% DSGE Parameters
beta = 0.99;     % Discount factor
sigma = 1;       % Risk aversion
phi = 1.5;       % Inflation response to output gap
rho = 0.9;       % Persistence of technology shock
sigma_epsilon = 0.01;  % Std. dev. of shock
% Time Horizon
T = 100;

% Initialize shock and output gap
epsilon = sigma_epsilon * randn(T,1);  % Random shocks
output_gap = zeros(T,1);

% Simulate the output gap under shocks
for t = 2:T
    output_gap(t) = rho * output_gap(t-1) + epsilon(t);
end

% Plot the output gap over time
figure;
plot(1:T, output_gap, 'LineWidth', 2);
xlabel('Time');
ylabel('Output Gap');
title('Macroeconomic Shock Simulation (Technology Shock)');
grid on;
% Operational risk sensitivity to macro shocks
sensitivity_factor = 0.5;  % How sensitive the firm is to output gap changes

% Operational Risk Losses
operational_losses = max(0, sensitivity_factor * output_gap * 1e5 + randn(T,1) * 1e4);

% Plot operational losses
figure;
plot(1:T, operational_losses, 'r-', 'LineWidth', 2);
xlabel('Time');
ylabel('Operational Loss ($)');
title('Operational Risk Losses Driven by Macroeconomic Shocks');
grid on;
% Stress Testing Operational Risk under Severe Shock
severe_shock = output_gap * 2;  % Amplified macro shock

% New operational losses under stress
stress_operational_losses = max(0, sensitivity_factor * severe_shock * 1e5 + randn(T,1) * 1e4);

% Plot comparison
figure;
plot(1:T, operational_losses, 'b-', 'LineWidth', 2);
hold on;
plot(1:T, stress_operational_losses, 'r--', 'LineWidth', 2);
xlabel('Time');
ylabel('Operational Loss ($)');
legend('Normal Scenario', 'Stress Scenario');
title('Stress Testing Operational Risk with DSGE Shocks');
grid on;
% --- Multi-Risk Integration with DSGE Shocks ---

% Credit Risk Parameters
credit_sensitivity = 0.4;  % Sensitivity to macro shocks
default_probability = max(0, credit_sensitivity * output_gap + randn(T,1) * 0.02);

% Market Risk Parameters
market_sensitivity = 0.3;
market_volatility = market_sensitivity * output_gap + randn(T,1) * 0.05;

% Aggregate Risk Losses
total_risk_loss = operational_losses + default_probability * 1e6 + abs(market_volatility * 5e5);

% Plot Total Risk Losses
figure;
plot(1:T, total_risk_loss, 'k-', 'LineWidth', 2);
xlabel('Time');
ylabel('Total Risk Loss ($)');
title('Aggregate Risk Losses: Credit, Market, and Operational Risks');
grid on;
% Correlation Matrix of Risks
risk_data = [default_probability, market_volatility, operational_losses];
correlation_matrix = corr(risk_data);

% Visualization of Correlation Matrix
figure;
heatmap({'Credit Risk', 'Market Risk', 'Operational Risk'}, ...
        {'Credit Risk', 'Market Risk', 'Operational Risk'}, ...
        correlation_matrix, 'Colormap', jet, 'ColorLimits', [-1 1]);
title('Correlation Matrix Between Risk Types');
% Value at Risk (95% Confidence)
VaR_95 = prctile(total_risk_loss, 5);

% Conditional Value at Risk (CVaR)
CVaR_95 = mean(total_risk_loss(total_risk_loss <= VaR_95));

% Display Results
fprintf('95%% Value-at-Risk (VaR): $%.2f\n', VaR_95);
fprintf('95%% Conditional Value-at-Risk (CVaR): $%.2f\n', CVaR_95);
% Define Stress Scenarios
severe_shock = output_gap * 3;  % Severe downturn
moderate_shock = output_gap * 2; % Moderate downturn

% Stress Testing Losses
severe_loss = max(0, sensitivity_factor * severe_shock * 1e5 + randn(T,1) * 1e4);
moderate_loss = max(0, sensitivity_factor * moderate_shock * 1e5 + randn(T,1) * 1e4);

% Plot Stress Testing Results
figure;
plot(1:T, operational_losses, 'b-', 'LineWidth', 2);
hold on;
plot(1:T, moderate_loss, 'm--', 'LineWidth', 2);
plot(1:T, severe_loss, 'r-.', 'LineWidth', 2);
xlabel('Time');
ylabel('Operational Loss ($)');
legend('Normal Scenario', 'Moderate Stress', 'Severe Stress');
title('Stress Testing with DSGE Scenarios');
grid on;
% Mitigation Recommendations
if VaR_95 > 2e6
    fprintf('Recommendation: Increase capital reserves to cover extreme losses.\n');
end

if correlation_matrix(1,3) > 0.7
    fprintf('Recommendation: Implement diversification strategies to reduce correlated risks.\n');
end

if mean(severe_loss) > mean(moderate_loss) * 1.5
    fprintf('Recommendation: Enhance operational risk controls under severe conditions.\n');
end
% DSGE Parameters
rho_A = 0.9;  % Productivity shock persistence
sigma_A = 0.02;  % Volatility of productivity shock

rho_m = 0.85;  % Monetary policy persistence
sigma_m = 0.01;  % Volatility of monetary policy shock

rho_f = 0.8;   % Fiscal policy persistence
sigma_f = 0.015;  % Volatility of fiscal policy shock

numSimulations = 10000;
% Initialize Shock Variables
A_shock = zeros(1, numSimulations);  % Productivity
m_shock = zeros(1, numSimulations);  % Monetary policy
f_shock = zeros(1, numSimulations);  % Fiscal policy

% Generate Shocks
for t = 2:numSimulations
    A_shock(t) = rho_A * A_shock(t-1) + sigma_A * randn;
    m_shock(t) = rho_m * m_shock(t-1) + sigma_m * randn;
    f_shock(t) = rho_f * f_shock(t-1) + sigma_f * randn;
end
% Risk Impact Adjustments
creditRiskImpact = 0.05 + 0.02 * m_shock;  % Higher rates → higher defaults
marketRiskImpact = 0.03 + 0.01 * A_shock; % Productivity drop → market fall
operationalRiskImpact = 0.02 + 0.015 * f_shock; % Fiscal stress → ops risk

% Total Loss Calculation
totalLosses = zeros(1, numSimulations);
for i = 1:numSimulations
    totalLosses(i) = creditRiskImpact(i) * 1e6 + ...   % Credit Risk Loss
                     marketRiskImpact(i) * 0.5e6 + ... % Market Risk Loss
                     operationalRiskImpact(i) * 0.3e6; % Operational Risk Loss
end
% Key Risk Metrics
expectedLoss = mean(totalLosses);
maxLoss = max(totalLosses);
VaR95 = prctile(totalLosses, 5);

fprintf('Expected Loss: $%.2f\n', expectedLoss);
fprintf('Maximum Loss: $%.2f\n', maxLoss);
fprintf('95%% Value-at-Risk (VaR): $%.2f\n', VaR95);
% Visualizing Total Loss Distribution
figure;
histogram(totalLosses, 50, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Total Loss ($)');
ylabel('Frequency');
title('Total Loss Distribution with DSGE Shocks');
grid on;


