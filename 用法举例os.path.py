import os

# 示例1：简单的文件路径
src_path1 = "audio_file.mp3"
base1, ext1 = os.path.splitext(src_path1)
print(f"源文件路径：{src_path1}")
print(f"文件名（不含扩展名）：{base1}")
print(f"扩展名：{ext1}")

# 示例2：包含目录的文件路径
src_path2 = "data/song.wav"
base2, ext2 = os.path.splitext(src_path2)#os.path.是一个模块函数
print(f"\n源文件路径：{src_path2}")
print(f"文件名（不含扩展名）：{base2}")
print(f"扩展名：{ext2}")
