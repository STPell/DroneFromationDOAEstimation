
f = open("drone_orientation_from_ripper/res_20220301-102126.csv", "r")
data = f.readlines()
f.close()

header = data[0]
data = data[1:]

f = open("drone_orientation_from_ripper/res_20220301-102126_corrected.csv", "w")
f.write(f"{header}")
for i in range(305):
    num_sectors, snr, num_repetitions, width, doa_rmse, std_dev = data[i].split(',')
    div_point = len(str(i))
    doa, rmse = doa_rmse[:div_point], doa_rmse[div_point:]
    print(doa, rmse)
    f.write(f"{num_sectors},{snr},{num_repetitions},{width},{doa},{rmse},{std_dev}")

f.close()
