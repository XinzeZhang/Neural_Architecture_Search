Write-Host "============================================================"
Write-Host "python SCSF.py --n_kernel 768 --k_allTrain_epochs  160 "
Write-Host "============================================================"
python SCSF.py --n_kernel 768 --k_allTrain_epochs 160 
Write-Host "============================================================"
Write-Host "python SCSF.py --n_kernel 768 --k_allTrain_epochs  130 "
Write-Host "============================================================"
python SCSF.py --n_kernel 768 --k_allTrain_epochs 130
Write-Host "============================================================"
Write-Host "python SCSF.py --n_kernel 768 --k_allTrain_epochs  140 "
Write-Host "============================================================"
python SCSF.py --n_kernel 768 --k_allTrain_epochs 140
Write-Host "============================================================"
Write-Host "python SCSF.py --n_kernel 768 --k_allTrain_epochs  170 "
Write-Host "============================================================"
python SCSF.py --n_kernel 768 --k_allTrain_epochs 170
# for($i=140; $i -le 300; $i=$i+10)
# {
#     Write-Host "============================================================"
#     Write-Host "python SCSF.py --n_kernel 768 --k_allTrain_epochs  $i "
#     Write-Host "============================================================"
#     python SCSF.py --n_kernel 768 --k_allTrain_epochs $i 
# }

