#!/usr/bin/python

from matplotlib import pyplot as plt

# SEQUENZIALE

plt.figure(figsize=(8,5))
words = ['sirpizza', '3kingdom', 'tyleisha', 'marumari', 'giacomix', 'dbcookie', 'Yessssss', 'Mypaypa1', '6Melissa', '1Mazzola']
times = [0.34943, 1.03047, 1.54878, 2.76967, 4.02124, 4.74491, 5.25437, 5.79248, 6.16216, 8.07845]
plt.plot(words, times)
plt.ylabel('Tempi versione sequenziali (s)')
plt.tight_layout()
plt.grid()
plt.savefig('tempi_seq.png')

# CUDA

plt.figure(figsize=(8,5))
words = ['3kingdom', 'giacomix', '6Melissa']
blocks = [8, 16, 32, 64, 128, 256, 512]

seq_3kingdom=1.03047
tempi_3kingdom = [2.68755, 1.33862, 0.67805, 0.64417, 0.64371, 0.64565, 0.67915]
speedup_3kingdom = [seq_3kingdom/i for i in tempi_3kingdom]

seq_giacomix=4.02124
tempi_giacomix = [2.68760, 1.34107, 0.67768, 0.64444, 0.64299, 0.64379, 0.67926]
speedup_giacomix = [seq_giacomix/i for i in tempi_giacomix]

seq_6Melissa=6.16216
tempi_6Melissa = [2.70067, 1.33751, 0.67724, 0.63960, 0.64287, 0.64340, 0.68120]
speedup_6Melissa = [seq_6Melissa/i for i in tempi_6Melissa]


plt.plot(blocks, speedup_3kingdom)
plt.plot(blocks, speedup_giacomix)
plt.plot(blocks, speedup_6Melissa)

plt.ylabel('Speedup versione parallela con CUDA')
plt.xlabel('Numero di blocchi e numero di thread per blocco')
plt.gca().legend(('3kingdom','giacomix', '6Melissa'), loc='upper right')
plt.tight_layout()
plt.grid()
plt.xscale('log', basex=2)
plt.savefig('tempi_cuda.png')

# OPENMP

plt.figure(figsize=(8,5))
words = ['3kingdom', 'giacomix', '6Melissa']
nThreads = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

seq_3kingdom=1.03047
tempi_3kingdom = [1.01924, 1.56739, 0.309739, 0.409104, 1.42765, 2.0312, 2.0312, 1.71969, 0.954556, 1.7443, 0.632172, 0.58017]
speedup_3kingdom = [seq_3kingdom/i for i in tempi_3kingdom]

seq_giacomix=4.02124
tempi_giacomix = [3.44626, 2.51188, 1.90975, 2.35197, 2.14367, 1.95578, 2.11623, 2.06844, 2.11548, 1.87124, 2.00013, 2.64471]
speedup_giacomix = [seq_giacomix/i for i in tempi_giacomix]

seq_6Melissa=6.16216
tempi_6Melissa = [2.90196, 1.31437, 1.76301, 1.89436, 1.36377, 0.506136, 0.619697, 0.530121, 1.85259, 1.98874, 1.52196, 1.62909]
speedup_6Melissa = [seq_6Melissa/i for i in tempi_6Melissa]


plt.plot(nThreads, speedup_3kingdom)
plt.plot(nThreads, speedup_giacomix)
plt.plot(nThreads, speedup_6Melissa)

plt.ylabel('Speedup versione parallela con OpenMP')
plt.xlabel('Numero di thread')
plt.gca().legend(('3kingdom','giacomix', '6Melissa'), loc='upper right')
plt.tight_layout()
plt.grid()
plt.xscale('log', basex=2)
plt.savefig('tempi_openmp.png')
