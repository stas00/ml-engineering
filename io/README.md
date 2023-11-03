# Filesystems and IO

## 3 ML IO needs

There are 3 distinct IO needs in the ML workload:

1. You need to be able to feed the DataLoader fast - (super fast read, don't care about fast write) - requires sustainable load for hours and days
2. You need to be able to write checkpoints fast - (super fast write, fastish read as you will be resuming a few times) - requires burst writing - you want super fast to not block the training for long (unless you use some sort of cpu offloading to quickly unblock the training)
3. You need to be able to load and maintain your codebase - (medium speed for both reading and writing) - this also needs to be shared since you want all nodes to see the same codebase - as it happens only during the start or resume it'll happen infrequently

As you can see these 3 have very different requirements both on speed and sustainable load, and thus ideally you'd have 3 different filesystems, each optimized for the required use case.

If you have infinite funds, of course, get a single super-fast read, super-fast write, that can do that for days non-stop. But for most of us, this is not possible so getting 2 or 3 different types of partitions where you end up paying much less is a wiser choice.


## Which Filesystem to choose

**The fastest solution is Parallel FileSystem**

- [Lustre FS](https://www.lustre.org/) (Open Source)
- [GPFS](https://en.wikipedia.org/wiki/GPFS) (IBM)

Both have been around for 2+ decades.

case study: At JeanZay HPC we were saving 2.3TB checkpoint in parallel on 384 processes in 40 secs! This is insanely fast - and it was GPFS over NVME drives.

More Parallel File Systems I have discovered recently:

- [BeeGFS](https://www.beegfs.io)


Most clouds provide at least one implementation of these, but not all. If your cloud provider doesn't provide one and they don't have a fast enough alternative you should reconsider.

**Slow solutions**

- [NFS](https://en.wikipedia.org/wiki/Network_File_System) - has been around for 4 decades but you don't want it as it's extremely slow. Especially since most providers have its very old version 3. It has a very long latency. It can take 30min to install several python packages and 20 seconds to load `import pytorch`. Its main problem is that it's very slow at handling meta-data so if you have a lot of small files, it just can't handle it well. And if you're into Python - it has thousands of small files. It's probably OK'ish if you have a few huge files. Otherwise, stay away for any serious ML work load.

case study: Python is so bad at having tens of thousand of tiny files that if you have many conda environments you are likely to run of inodes. At JeanZay HPC we had to ask for a special small partition where we would install all conda environments because we kept running out of inodes on normal GPFS partitions.

**OK'ish solutions**

There are many other OK'ish solutions offered by various clouds. Benchmark those seriously before you commit to any.


## Local storage beats cloud storage

While cloud storage is cheaper the whole idea of fetching and processing your training data stream dynamically at training time is very problematic with a huge number of issues around it.

Same goes for dynamic offloading of checkpoints to the cloud.

It's so much better to have enough disc space locally for data loading.

For checkpointing there should be enough local disc space for saving a checkpoint in a fast and reliable way and then having a crontab job or a slurm job to offload it to the cloud. Always keep the last few checkpoints locally for a quick resume, should your job crash, as it'd be very expensive to wait to fetch the checkpoint from the cloud for a resume.

case study: we didn't have a choice and had to use cloud storage for dataloading during IDEFICS-80B training as we had barely any local storage and since it was multimodal data it was many TBs of data. We spent many weeks trying to make this solution robust and it sucked at the end. The biggest issue was that it was very difficult at the time to keep track of RNG state for the DataSampler because the solution we used, well, didn't bother to take care of it. So a lot of data that took a lot of time to create was wasted (not used) and a lot of data was repeated, so we didn't have a single epoch of unique data.


## Beware that you're being sold only 80% of the storage you pay for

There is a subtle problem with distributed shared storage used on compute nodes. Since most physical disks used to build the large filesystems are only 0.5-2TB large, any of these physical disks can get full before the combined storage gets full. And thus they require constant rebalancing so that there will be no situation where one disk is 99% full and others are only 50% full. Since rebalancing is a costly operation, like most programming languages' garbage collection, it happens infrequently. And so if you run `df` and it reports 90% full, it's very likely that any of the programs can fail at any given time.

From talking to IO engineers, the accepted reality (that for some reason is not being communicated to customers) is that only about 80% of distributed large storage is reliable.

Which means that if you want to have 100TB of reliable cloud storage you actually need to buy 125TB of storage, since 80% of that will be 100TB. So you need to plan to pay 25% more than what you provisioned for your actual needs. I'm not sure why the customer should pay for the technology deficiency but that's how it is.

## Don't forget the checksums

When you sync data to and from the cloud make sure to research whether the tool you use checks the checksums, otherwise you may end up with corrupt during transmission data. Some tools do it automatically, others you have to enable this feature (since it usually comes at additional compute cost and transmission slowdown). Better slow, but safe.

These are typically MD5 and SHA256 checksums. Usually MD5 is sufficient if your environment is safe, but if you want the additional security do SHA256 checksums.

## Benchmarks

- [HPC IO Benchmark Repository](https://github.com/hpc/ior)

XXX: expand on how it's used
