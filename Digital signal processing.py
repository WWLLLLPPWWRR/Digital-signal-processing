import matplotlib.pyplot as plt
import pandas as pd
import openpyxl
import math

# Visualisation with matplotlib


def customGraph(numbers: list):
    plt.plot(numbers)
    plt.show()


# Reading signal sequences


class DealingWithFiles:
    def readTxt(self, file):
        signal = []
        with open(file=file) as f:
            for i in f:
                signal.append(float(i))
            print(signal)
        # Visualization
        # customGraph(signal)

    def readExcel(self, file):
        signal = pd.read_excel(file)
        signal = list(signal['s(t)'])
        sl = list(map(float, signal))
        print(sl)
        # Visualization
        # customGraph(sl)

# Processing


class ProcessingProcedures:
    def correlation(self, f: list, g: list):
        border = max(len(g), len(f))
        result = []
        for tau in range(-1, border):
            sum_for_shift = 0
            for s in range(-1, border):
                if tau + s > len(f) - 1 or tau + s < 0:
                    par1 = 0
                else:
                    par1 = f[s + tau]

                if s > len(g) - 1 or s < 0:
                    par2 = 0
                else:
                    par2 = g[s]
                print(par1, par2)
                sum_for_shift += par1 * par2
            result.append(sum_for_shift)
        print(result)

    def fourier_transform(self, signal: list, time: list, freq_sample: float):

        plt.plot(time, signal)
        plt.title('Signal')
        plt.show()

        time_sample = time[1] - time[0]
        # Reasonable number of harmonics according to Kotelnikov's (Nyquist's) Theorem
        number_of_harmonics = math.floor( ((1 / (2 * time_sample)) / freq_sample) / 2 ) + 1
        list_of_a_coe = []
        list_of_b_coe = []
        frequencies = []
        for harmonic in range(number_of_harmonics):
            # For each single time sample multiplies with constants under COS or SIN and every signal sample
            # multiplies with that constant
            for_a = list(map(lambda n, t: n * math.cos(2 * math.pi * freq_sample * harmonic * t), signal, time))
            for_b = list(map(lambda n, t: n * math.sin(2 * math.pi * freq_sample * harmonic * t), signal, time))
            a_coe = (2 / len(signal)) * sum(for_a)
            b_coe = (2 / len(signal)) * sum(for_b)
            list_of_a_coe.append(a_coe)
            list_of_b_coe.append(b_coe)
            frequencies.append(freq_sample*harmonic)

        # Amplitude characteristic
        amps = list(map(lambda a, b: math.sqrt(a**2 + b**2), list_of_a_coe, list_of_b_coe))
        plt.plot(frequencies, amps)
        plt.title('Amplitude characteristic')
        plt.show()

        # Phase characteristic
        phases = list(map(lambda a, b: - math.atan(b / a), list_of_a_coe, list_of_b_coe))
        plt.plot(frequencies, phases)
        plt.title('Phase characteristic')
        plt.show()

        # Harmonics
        def harmonic_visualisation(coe_a, coe_b, frequency, time_line):
            one_harmonic = list(map(lambda smpl: coe_a * math.cos(smpl*frequency*2*math.pi) + coe_b * math.sin(smpl*frequency*2*math.pi), time_line))
            plt.plot(time, one_harmonic)

        for i in range(len(frequencies)):
            harmonic_visualisation(list_of_a_coe[i], list_of_b_coe[i], frequencies[i], time)
        plt.show()

        return amps, phases, frequencies, signal

    def reverse_fourier_transform(self, timeline: list, frequencies: list, amplitudes: list, phases: list):
        harmonics = []
        for i in range(len(frequencies)):
            harmonic = list(map(lambda t: (amplitudes[i] * math.cos(2*math.pi*frequencies[i]*t - phases[i])), timeline))
            harmonics.append(harmonic)
        reversed_signal = [0] * len(timeline)
        half_a0 = sum(amplitudes) / (2*len(frequencies))
        for j in range(len(timeline)):
            for k in range(len(harmonics)):
                reversed_signal[j] += harmonics[k][j]

        reversed_signal = list(map(lambda x: x + half_a0, reversed_signal))
        plt.plot(timeline, reversed_signal)
        plt.title('Recovered signal')
        plt.show()

        return timeline, reversed_signal

    def convolution(self, f: list, g: list):
        result = []
        # Shift
        for tau in range(len(g)+len(f)-1):
            sum_for_shift = 0
            for t in range(len(f)):
                if tau-t > len(g)-1 or tau-t < 0:
                    par1 = 0
                else:
                    par1 = g[tau-t]
                sum_for_shift += par1 * f[t]
            result.append(sum_for_shift)
        return result

    def spectrum_convolution(self, signal1, timeline1, signal2, timeline2, freq_sample):
        amps1, phases1, frequencies1, signal1 = self.fourier_transform(signal1, timeline1, freq_sample)
        amps2, phases2, frequencies2, signal2 = self.fourier_transform(signal2, timeline2, freq_sample)

        common_amp = list(map(lambda x, y: x * y, amps1, amps2))
        common_phase = list(map(lambda x, y: x - y, phases1, phases2))

        timeline, result_signal = self.reverse_fourier_transform(timeline1, frequencies1, common_amp, common_phase)
        return timeline, result_signal

    def spectrum_deconvolution(self, signal1, timeline1, signal2, timeline2, freq_sample):
        amps1, phases1, frequencies1, signal1 = self.fourier_transform(signal1, timeline1, freq_sample)
        amps2, phases2, frequencies2, signal2 = self.fourier_transform(signal2, timeline2, freq_sample)

        common_amp = list(map(lambda x, y: x / y, amps1, amps2))
        common_phase = list(map(lambda x, y: x + y, phases1, phases2))

        timeline, result_signal = self.reverse_fourier_transform(timeline1, frequencies1, common_amp, common_phase)
        return timeline, result_signal

    def deconvolution(self):
        pass

    def zTransformation(self, t: list):
        pass

    def fast_fourier_transform(self, t: list):
        pass

    def lowFilter(self, t: list):
        pass

    def spectrumFiltration(self, t: list):
        pass

    def laplas(self, signal: list):
        pass

    def wavelet(self):
        pass


# Signal and time examples

def ts(delta_t, number_of_samples):
    time_samples = [delta_t*i for i in range(number_of_samples)]
    print(len(time_samples))
    return time_samples


tsts = ts(0.002, 28)
env = [-0.5, -0.2, 0.2857, -0.25]
env_time = [0.0167, 0.0278, 0.0426, 0.0533]
trtr = [4.0, 8.0, 4.0, 2.0, 0.0, -8.0, -12.0, -14.0, -12.0, -10.0, -6.0, 0.0, 20.0, 23.0, 16.0, 14.0, 10.0, 23.0, 18.0, 9.0, 0.0, -4.0, -8.0, -10.0, -12.0, -16.0, -14.0, -9.0]


# Statistics - part for the future. . .


