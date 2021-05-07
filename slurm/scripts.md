
## Convert SLURM_JOB_NODELIST into a hostfile

```
# autogenerate the hostfile for deepspeed
# 1. deals with: SLURM_JOB_NODELIST in either of 2 formats:
# r10i1n8,r10i2n0
# r10i1n[7-8]
# 2. and relies on SLURM_STEP_GPUS=0,1,2... to get how many gpu slots per node
#
# usage:
# makehostfile > hostfile
function makehostfile() {
perl -le '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"}; $_=$ENV{"SLURM_JOB_NODELIST"};
if (/^(.*?)\[(\d)+-(\d+)\]/) { print map { "$1$_ slots=$slots\n" } $2..$3} 
elsif (/,/) { print map { "$1$_ slots=$slots\n" } split /,/ } '
}
```
