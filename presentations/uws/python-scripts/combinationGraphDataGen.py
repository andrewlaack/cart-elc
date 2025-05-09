import math

nVals = [100, 200, 300, 400, 500]
rVals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

results = []

for r in rVals:
    results.append([])
    for n in nVals:
        # With m = r, math.comb(m, r) = math.comb(r, r) = 1.
        # Therefore, the complexity is:
        # math.comb(n, r) * (r^3 + r * n)
        result = math.comb(n, r) * (r**3 + r*n)
        results[-1].append(str(result))

for res in rVals:
    if res == 0:
        continue  # Skip r = 0 since it gives 0 instructions.
    print(str(res) + " & ", end="")
    itr = 1
    for r_val in results[res]:
        if itr == len(nVals):
            print(r_val + " \\\\")
        else:
            print(r_val + " &", end=" ")
        itr += 1

