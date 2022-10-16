% A simple OFDM simulator for integrated sensing and communication systems
% Source code: https://github.com/YongzhiWu/OFDM_ISAC_simulator
clc;
clear;

%% ISAC Transmitter

% System Parameters
c0 = physconst('LightSpeed'); % light of speed
fc = 30e9; % carrier frequency
lambda = c0 / fc; % wavelength
N = 256; % number of subcarriers
M = 16; % number of symbols
delta_f = 15e3 * 2^6; % subcarrier spacing
T = 1 / delta_f; % symbol duration
Tcp = T / 4; % cyclic prefix duration
Ts = T + Tcp; % total symbol duration
CPsize = N / 4; % cyclic prefix length
bitsPerSymbol = 2; % bits per symbol
qam = 2^(bitsPerSymbol); % 4-QAM modulation

% Transmit data
data = randi([0 qam - 1], N, M);
TxData = qammod(data, qam, 'UnitAveragePower', true);

% OFDM modulator
TxSignal = ifft(TxData, N); % IFFT
TxSignal_cp = [TxSignal(N - CPsize + 1: N, :); TxSignal]; % add CP
TxSignal_cp = reshape(TxSignal_cp, [], 1); % time-domain transmit signal

%% Communication channel
PowerdB = [0 -8 -17 -21 -25]; % Channel tap power profile [dB]
Delay = [0 3 5 6 8]; % Channel delay sample
Power = 10.^(PowerdB/10); % Channel tap power profile
Ntap = length(PowerdB); % Chanel tap number
Lch = Delay(end)+1; % Channel length
channel=(randn(1,Ntap)+1j*randn(1,Ntap)).*sqrt(Power/2); % Rayleigh fading
h=zeros(1,Lch); h(Delay+1) = channel;
ComSNRdB = 15; % SNR of communication channel
RxSignal = conv(TxSignal_cp, h);
RxSignal = RxSignal(1:end - Lch + 1);
RxSignal = awgn(RxSignal, ComSNRdB, 'measured'); % add AWGN

%% Communication receiver
RxSignal = reshape(RxSignal, [N + CPsize, M]);
RxSignal_remove_cp = RxSignal(CPsize + 1: N + CPsize, :); % remove CP
RxData = fft(RxSignal_remove_cp, N); % FFT

H_channel = fft([h zeros(1, N-Lch)]).'; % perfect channel estimation
H_channel = repmat(H_channel, [1, M]);

% MMSE equalization
C = conj(H_channel)./(conj(H_channel).*H_channel + 10^(-ComSNRdB/10));
demodRxData = RxData .* C;
demodRxData = qamdemod(demodRxData, qam, 'UnitAveragePower', true);
errorCount = sum(sum(de2bi(demodRxData, bitsPerSymbol) ~= de2bi(data, bitsPerSymbol)));
comResult = ['Number of error bits: ', num2str(errorCount)];
disp(comResult);

%% Radar channel
target_pos = 30; % target distance
target_delay = range2time(target_pos, c0);
target_speed = 20; % target velocity
target_dop = speed2dop(2 * target_speed, lambda);
RadarSNRdB = 5; % SNR of radar sensing channel
RadarSNR = 10.^(RadarSNRdB/10);

%% Radar receiver
% Received data in the frequency domain
RxData = zeros(size(TxData));
for kSubcarrier = 1:N
    for mSymbol = 1:M
        RxData(kSubcarrier, mSymbol) = sum(sqrt(RadarSNR) * TxData(kSubcarrier, mSymbol) .* exp(-1j * 2 * pi * fc * target_delay)  .* exp(1j * 2 * pi * mSymbol *...
            Ts .* target_dop) .* exp(-1j * 2 * pi * kSubcarrier .* target_delay *...
            delta_f) ) + sqrt(1/2)* (randn() +1j * randn());
    end
end

% Radar sensing algorithm (FFT)
dividerArray = RxData ./ TxData;
NPer = 16 * N;
normalizedPower = abs(ifft(dividerArray, NPer, 1));
normalizedPower_dB = 10 * log10(normalizedPower);
mean_normalizedPower = mean(normalizedPower, 2);
mean_normalizedPower = mean_normalizedPower / max(mean_normalizedPower);
mean_normalizedPower_dB = 10 * log10(mean_normalizedPower);
[~, rangeEstimation] = max(mean_normalizedPower_dB);
distanceE = rangeEstimation * c0 / (2 * delta_f * NPer); % estimated target range
sensingResult = ['The estimated target range is ', num2str(distanceE), ' m.'];
disp(sensingResult);