#!/usr/bin/python

from matplotlib import pyplot as plt

# SEQUENZIALE vs ALTRI

plt.figure(figsize=(8,5))
words = ['sirpizza', '3kingdom', 'tyleisha', 'marumari', 'giacomix', 'dbcookie', 'Yessssss', 'Mypaypa1', '6Melissa', '1Mazzola']
times_seq = [0.231341, 0.740162, 1.099185, 2.057434, 2.651718, 3.041978, 3.701794, 3.890734, 4.635460, 5.194617]
times_cuda = [0.681015, 0.641720, 0.644346, 0.645845, 0.643087, 0.643216, 0.646316, 0.647170, 0.645394, 0.647059] # 128 blocchi e thread per blocco
times_openmp = [0.832012, 1.07067, 1.35734, 0.472368, 2.20688, 0.343134, 1.56616, 0.419804, 1.22991, 1.82078] # 32 thread
plt.plot(words, times_seq)
plt.plot(words, times_cuda)
plt.plot(words, times_openmp)
plt.ylabel('Tempo (s)')
plt.gca().legend(('Sequenziale','CUDA (128 thread/blocco)', 'OpenMP (32 Thread)'), loc='upper left')
plt.tight_layout()
plt.grid()
plt.savefig('tempi.png')

# CUDA

plt.figure(figsize=(8,5))
words = ['3kingdom', 'giacomix', '6Melissa']
blocks = [8, 16, 32, 64, 128, 256, 512]

seq_3kingdom=times_seq[1]
tempi_3kingdom = [8.628553, 4.307805, 2.062298, 1.095152, 0.646086, 0.647663, 0.650296]
speedup_3kingdom = [seq_3kingdom/i for i in tempi_3kingdom]

seq_giacomix=times_seq[4]
tempi_giacomix = [8.631157, 4.322363, 2.067904, 1.096998, 0.646703, 0.649602, 0.652175]
speedup_giacomix = [seq_giacomix/i for i in tempi_giacomix]

seq_6Melissa=times_seq[8]
tempi_6Melissa = [8.625993, 4.316520, 2.065404, 1.095974, 0.646805, 0.650377, 0.651141]
speedup_6Melissa = [seq_6Melissa/i for i in tempi_6Melissa]

plt.plot(blocks, speedup_3kingdom)
plt.plot(blocks, speedup_giacomix)
plt.plot(blocks, speedup_6Melissa)

plt.ylabel('Speedup CUDA')
plt.xlabel('Numero di thread per blocco')
plt.gca().legend(('3kingdom','giacomix', '6Melissa'), loc='upper left')
plt.tight_layout()
plt.grid()
plt.xscale('log', basex=2)
plt.savefig('tempi_cuda.png')

# OPENMP

plt.figure(figsize=(8,5))
words = ['3kingdom', 'giacomix', '6Melissa']
nThreads = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

tempi_3kingdom = [0.666503, 1.19207, 0.285334, 0.661375, 1.25394, 2.00221, 1.84419, 1.69569, 0.808561, 1.84429, 0.88455, 1.22803]
speedup_3kingdom = [seq_3kingdom/i for i in tempi_3kingdom]

tempi_giacomix = [2.38582, 2.0866, 2.27441, 2.22935, 2.07361, 2.16398, 2.22033, 2.09397, 1.94362, 1.99642, 1.482312, 1.33628]
speedup_giacomix = [seq_giacomix/i for i in tempi_giacomix]

tempi_6Melissa = [1.74243, 0.981246, 2.2642, 1.96153, 1.74995, 0.408785, 0.533524, 1.09762, 2.00256, 1.98084, 1.60031, 1.03923]
speedup_6Melissa = [seq_6Melissa/i for i in tempi_6Melissa]

plt.plot(nThreads, speedup_3kingdom)
plt.plot(nThreads, speedup_giacomix)
plt.plot(nThreads, speedup_6Melissa)

plt.ylabel('Speedup OpenMP')
plt.xlabel('Numero di thread')
plt.gca().legend(('3kingdom','giacomix', '6Melissa'), loc='upper right')
plt.tight_layout()
plt.grid()
plt.xscale('log', basex=2)
plt.savefig('tempi_openmp.png')
