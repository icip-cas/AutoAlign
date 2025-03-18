import os
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
sd1 = script_dir+"/sc/sc1.sh"
sd2 = script_dir+"/sc/sc2.sh"
sd3 = script_dir+"/sc/sc3.sh"
sc_list = [sd1, sd2, sd3]

total_turns = 3
for sc in sc_list:
    os.chmod(sc, 0o755)
for run in range(1, total_turns+1):
    print(f"Running run {run}/{total_turns}")
    for path in sc_list:
        try:
            result = subprocess.run([path], capture_output=True, text=True, shell=True)
            print(f"Output: {result.stdout}")
            # print(f"Error: {result.stderr}")

        except Exception as e:
            print(e)

    



# # 获取返回码
# returncode = proc.wait()
# print("Return Code:", returncode)
