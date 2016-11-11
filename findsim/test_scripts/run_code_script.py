import os
import subprocess


# Change directory to the code directory
os.chdir("/home/gauravnanda/Desktop/cmpe239-DataMining/my_git_code/DataMining/findsim/build/")
print ("Current working director: " + os.getcwd())


# Define various test run parameters
eps_vals = [0.5, 0.7, 0.9]
k_vals = [10, 50, 100]
mod_vals = ['ij', 'dynamic']

input_file_names = ['data/wiki1k.csr', 'data/wiki1.csr', 'data/wiki2.csr']
output_file_name = ('%s.nbrs.%s.%s.%s.csr')

cmd = ('./findsim -eps %f -k %d %s %s --mode %s')


def parse_time(result):
    time = result.split("Similarity search:  ")[1]
    time_sec = time.split(' ')[0]
    return float(time_sec)

# Define file to write output to
f = open("/home/gauravnanda/Desktop/saari_output", "w")
lines = []

for ip_file in input_file_names:
    lines.append("Data Set: " + str(ip_file))
    print ("Data Set: " + str(ip_file))
    for mode in mod_vals:
        lines.append("Mode = " + str(mode))
        print ("Mode = " + str(mode))
        for k in k_vals:
            for eps in eps_vals:
                avg_time = 0
                for i in range(3):
                    # Execute command
                    cmd_exec = cmd % (eps, k, ip_file, "xxx", mode)
                    result_str = subprocess.check_output(cmd_exec, shell=True)
                    avg_time += parse_time(result_str)
                avg_time /= 3
                lines.append("k= " + str(k) + " eps= "  + str(eps) + " === " + str(avg_time))
                print ("mode= " + str(mode) + " k= " + str(k) + " eps= "  + str(eps) + " === " + str(avg_time))

# Write results to file
f.write("\n".join(lines))
f.close()
print ("Written to file")

