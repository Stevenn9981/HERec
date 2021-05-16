nohup python -u HERec_spl.py --um 1 --im 1 > 1+1.txt 2>&1 &
nohup python -u HERec_spl.py --um 2 --im 1 > 2+1.txt 2>&1 &
nohup python -u HERec_spl.py --um 3 --im 1 > 3+1.txt 2>&1 &
nohup python -u HERec_spl.py --um 4 --im 1 > 4+1.txt 2>&1 &
python -u HERec_spl.py --um 1 --im 2 > 1+2.txt
nohup python -u HERec_spl.py --um 2 --im 2 > 2+2.txt 2>&1 &
nohup python -u HERec_spl.py --um 3 --im 2 > 3+2.txt 2>&1 &
nohup python -u HERec_spl.py --um 4 --im 2 > 4+2.txt 2>&1 &
python -u HERec_spl.py --um 1 --im 3 > 1+3.txt
nohup python -u HERec_spl.py --um 2 --im 3 > 2+3.txt 2>&1 &
nohup python -u HERec_spl.py --um 3 --im 3 > 3+3.txt 2>&1 &
nohup python -u HERec_spl.py --um 4 --im 3 > 4+3.txt 2>&1 &
