#!/bin/bash

# 记录开始时间（秒）
start=$(date +%s)

# 运行程序
python ../POVME-main/POVME2.py sample_POVME_input.ini

# 记录结束时间（秒）
end=$(date +%s)

# 计算耗时并输出
runtime=$((end - start))
echo "POVME运行总时间: ${runtime} 秒"
