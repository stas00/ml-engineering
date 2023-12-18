# SLURM Administration


## Run a command on multiple nodes

1. to avoid being prompted with:
```
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```
for every new node you haven't logged into yet, you can disable this check with:
```
echo "Host *" >> ~/.ssh/config
echo "  StrictHostKeyChecking no" >> ~/.ssh/config
```

Of course, check if that's secure enough for your needs. I'm making an assumption that you're already on the SLURM cluster and you're not ssh'ing outside of your cluster. You can choose not to set this and then you will have to manually approve each new node.

2. Install `pdsh`

You can now run the wanted command on multiple nodes.

For example, let's run `date`:

```
$ PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26] date
node-25: Sat Oct 14 02:10:01 UTC 2023
node-21: Sat Oct 14 02:10:02 UTC 2023
node-23: Sat Oct 14 02:10:02 UTC 2023
node-24: Sat Oct 14 02:10:02 UTC 2023
node-26: Sat Oct 14 02:10:02 UTC 2023
```

Let's do something more useful and complex. Let's kill all GPU-tied processes that didn't exit when the SLURM job was cancelled:

First, this command will give us all process ids that tie up the GPUs:

```
nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq
```

So we can now kill all those processes in one swoop:

```
 PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26]  "nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq | xargs -n1 sudo kill -9"
```



## Changing the state of the node

The change is performed by `scontrol update`

Examples:

To undrain a node that is ready to be used:
```
scontrol update nodename=node-5 state=idle
```

To remove a node from the SLURM's pool:
```
scontrol update nodename=node-5 state=drain
```


## Undrain nodes killed due to slow process exit

Sometimes processes are slow to exit when a job has been cancelled. If the SLURM was configured not to wait forever it'll automatically drain such nodes. But there is no reason for those nodes to not be available to the users.

So here is how to automate it.

The keys is to get the list of nodes that are drained due to `"Kill task failed"`, which is retrieved with:

```
sinfo -R | grep "Kill task failed"
```

now extract and expand the list of nodes, check that the nodes are indeed user-process free (or try to kill them first) and then undrain them.

Earlier you learned how to [run a command on multiple nodes](run-a-command-on-multiple-nodes) which we will use in this script.

Here is the script that does all that work for you: [undrain-good-nodes.sh](./undrain-good-nodes.sh)

Now you can just run this script and any nodes that are basically ready to serve but are currently drained will be switched to `idle` state and become available for the users to be used.


## Modify a job's timelimit

To set a new timelimit on a job, e.g., 2 days:
```
scontrol update JobID=$SLURM_JOB_ID TimeLimit=2-00:00:00
```

To add additional time to the previous setting, e.g. 3 more hours.
```
scontrol update JobID=$SLURM_JOB_ID TimeLimit=+10:00:00
```
