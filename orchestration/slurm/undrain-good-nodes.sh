#!/bin/bash

# When nodes get auto placed in drain because SLURM fails to wait till all last job's processes are killed and it just takes longer for them to finish, this script automatically checks if all processes tied to the gpu have been killed and if this is so it'll undrain those nodes

# get the nodes that were put to `drain` because the job was too slow to exit
nodes=( $(sinfo -R | grep "Kill task failed" | perl -lne '/(node-.*[\d\]]+)/ && print $1' | xargs -n1 scontrol show hostnames) )

good=()
bad=()

# declare an array called array and define 3 values
for n in "${nodes[@]}"; do
	echo "*** checking $n"

    # check if any processes are still stuck - when none there should be no output
    output=$(PDSH_RCMD_TYPE=ssh pdsh -w $n "nvidia-smi --query-compute-apps=pid --format=csv,noheader")
    if [ -z "$output" ]; then
        clean=1
    else
        clean=0
        # if there are processes running still try to kill them again and recheck if it was successful

        # kill any processes tying up the gpus
        PDSH_RCMD_TYPE=ssh pdsh -w $n "nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq | xargs -n1 sudo kill -9"

        echo "sleeping for 3 secs to let the processes exit"
        sleep 3

        # check if any processes are still stuck - when none there should be no output
        output=$(PDSH_RCMD_TYPE=ssh pdsh -w $n "nvidia-smi --query-compute-apps=pid --format=csv,noheader")
        if [ -z "$output" ]; then
            clean=1
        fi
    fi

    if [ $clean == 1 ]; then
        echo "no gpu processes are tied, undraining $n"
        sudo scontrol update NodeName=$n State=idle Reason="undrained by $USER"
        good+=($n)
    else
        echo "failed to kill all processed tied to gpus on $n"
        echo "ssh into $n and manually check the state of the node"
        bad+=($n)
    fi
    echo ""
done
