    Write-Host "============================================================"
Write-Host "python SCSF.py --n_kernel 768 --k_allTrain_epochs  1 "
Write-Host "============================================================"
python SCSF.py --n_kernel 768 --k_allTrain_epochs 1 --gpu 1
Write-Host "============================================================"
Write-Host "python SCSF.py --n_kernel 768 --k_allTrain_epochs  3 "
Write-Host "============================================================"
python SCSF.py --n_kernel 768 --k_allTrain_epochs 5 --gpu 1

for($i=0; $i -le 300; $i=$i+10)
{
    Write-Host "============================================================"
    Write-Host "python SCSF.py --n_kernel 768 --k_allTrain_epochs  $i "
    Write-Host "============================================================"
    python SCSF.py --n_kernel 768 --k_allTrain_epochs $i --gpu 1
}

