import matplotlib.pyplot as plt

q_1 = "2.68690907e-03 2.44155996e-02 5.61814916e-02 7.05630467e-02 5.27800338e-44 3.57397249e-02 2.86703288e-02 7.19722177e-02 3.01597016e-77 5.36750448e-03 8.10105557e-03 7.67814864e-02 3.77034461e-02 1.06302772e-01 1.97488335e-37 3.78998182e-02 1.63643429e-03 1.04731795e-01 1.10295671e-01 1.45278252e-01 2.95476659e-53 1.58406840e-02 2.59865766e-02 5.49841922e-03 2.74266387e-02 9.16403204e-04 3.72566552e-06"
q_2 = "1.35228401e-001 9.04593772e-103 1.16058529e-004 4.98924067e-047 2.13952678e-001 7.94705156e-157 4.41039135e-013 1.18671928e-003 1.17979619e-001 1.17041688e-190 7.63384309e-004 1.10217321e-008 2.00113584e-093 5.16477529e-069 1.32514817e-001 3.22356645e-020 2.82392379e-190 4.84952144e-072 9.32365517e-047 7.98532451e-003 4.63224528e-002 6.88822682e-178 2.02983607e-135 2.25017322e-118 6.52847504e-019 2.02058637e-203 3.43950534e-001"
alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '#']

q_1 = q_1.split()
q_1 = [float(i) for i in q_1]

q_2 = q_2.split()
q_2 = [float(i) for i in q_2]

plt.figure(1)
plt.bar(alphabets, q_1)
plt.xlabel('Alphabet')
plt.ylabel('Emission Probability')
plt.title('Emission Probability bar plot of state 1')
plt.savefig("state1.pdf")

plt.figure(2)
plt.bar(alphabets, q_2, color='g')
plt.xlabel('Alphabet')
plt.ylabel('Emission Probability')
plt.title('Emission Probability bar plot of state 2')
plt.savefig("state2.pdf")


