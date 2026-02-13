# Kubernetes

In general IMHO kubernetes is the wrong environment for doing training work, with [SLURM](../slurm/) being a much more user-friendly orchestration. SLURM builds on top of Unix and so it just makes multi-node coordination easier. But k8s reinvents the wheel completely, leading to lots of complexities and many normal Unix features are missing from its feature set, making it into a very challenging environment to work in, requiring a large support team to just deal with so many k8s problems and a lack of rudimentary Unix features (e.g., you can't even have Unix usernames and all users run under the same Unix username - think accountability and security!).

When forced into k8s one will find a way to get their work done, but at a huge human and $$ cost to the training shop.

This is going to be a small chapter since I can't imagine how I'd even approach covering k8s complexities here, so I'm just going to cover important gotchas that should ease your work a bit.

## Setup

### Overcoming job reset on CPU OOM event

This default feature enabled in k8s v1.28 makes absolutely no sense in the context of interactive training jobs. If any of your processes get CPU OOM'ed you get kicked out and your job gets reset. This feature is good for serving inference in production, but it's a huge problem for interactive training work, for example when one is trying to find out an optimal performance training configuration, as instead of being able to recover from a crashed process, adjust things and try again, you have to start from scratch and not even have logs, leading to the problem unless you manually sync those in time.

So to overcome this you need to get k8s to set `memory.oom.group = 0` either on the node pool level or cluster level configs - the default is `memory.oom.group = 1` so if your job gets killed on a cpu-oom event ask your k8s admin to make this change. The value of `0` will just kill the process that caused cpu-oom.

Here is how this is done. Kubernetes 1.32 introduced the kubelet flag [`singleProcessOOMKill`](https://github.com/kubernetes/kubernetes/pull/126096), which allows you to set `memory.oom.group = 0`.

```
        compute:
          additionalNodePools:
          	kubeletConfig: # this flag is added to node pool config
            - name: foo
                singleProcessOOMKill: true
```


To check the actual setting from within the running node, do:
```bash
$ cat /sys/fs/cgroup/memory.oom.group
```