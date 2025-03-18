import time

# 源文件路径和目标文件路径
source_file = "/home/wangjunxiang2025/AutoAlign/ui/temp_log/log/output.log"
target_file = "/home/wangjunxiang2025/AutoAlign/ui/temp_log/log/generate_log.log"

# 模拟日志生成
def simulate_logs(sleep_interval=0.5):  # 默认每 0.1 秒写入一行
    try:
        # 打开源文件
        with open(source_file, "r") as src:
            lines = src.readlines()  # 读取所有行

        # 打开目标文件，准备写入
        with open(target_file, "w") as tgt:
            for line in lines:
                tgt.write(line)  # 写入一行
                tgt.flush()  # 确保立即写入文件
                print(f"Wrote: {line.strip()}")  # 打印到控制台
                time.sleep(sleep_interval)  # 调整写入速度

        print("Simulation complete!")
    except FileNotFoundError:
        print(f"Error: File {source_file} not found!")
    except Exception as e:
        print(f"An error occurred: {e}")

# 运行模拟
if __name__ == "__main__":
    # 设置 sleep_interval 参数，例如 0.1 秒（10 行/秒）
    simulate_logs(sleep_interval=0.1)