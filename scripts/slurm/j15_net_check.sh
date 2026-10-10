#!/bin/bash
#SBATCH -J acc-j15-net
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=dcu:1
#SBATCH -t 00:05:00
#SBATCH -o /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j15-%j.out
#SBATCH -e /public/home/ac6ksiqdok/zksg-all/中科曙光/验收/logs/j15-%j.err

# J15：计算节点能否联网（决定 pip 装还是离线装）
echo "########## J15 on $(hostname) $(date) ##########"
for u in https://pypi.org/simple/ https://download.sourcefind.cn:65024/ https://files.pythonhosted.org/; do
  printf "%-45s " "$u"
  timeout 15 curl -sk -o /dev/null -w "HTTP %{http_code}  %{time_total}s\n" "$u" 2>&1 || echo "FAIL"
done
echo "--- pip 是否存在/可用 ---"
PY=/usr/local/python3.12/bin/python3
$PY -m pip --version 2>&1 | head -2
echo "########################################## J15 done ##########"
