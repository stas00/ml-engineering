# Filesystems and IO

## 3 ML IO needs

There are 3 distinct IO needs in the ML workload:

1. You need to be able to feed the DataLoader fast - (super fast read, don't care about fast write) - requires sustainable load for hours and days
2. You need to be able to write checkpoints fast - (super fast write, fastish read as you will be resuming a few times) - requires burst writing - you want super fast to not block the training for long (unless you use some sort of cpu offloading to quickly unblock the training)
3. You need to be able to load and maintain your codebase - (medium speed for both reading and writing) - this also needs to be shared since you want all nodes to see the same codebase - as it happens only during the start or resume it'll happen infrequently

As you can see these 3 have very different requirements both on speed and sustainable load, and thus ideally you'd have 3 different filesystems, each optimized for the required use case.

If you have infinite funds, of course, get a single super-fast read, super-fast write, that can do that for days non-stop. But for most of us, this is not possible so getting 2 or 3 different types of partitions where you end up paying much less is a wiser choice.

Incoming suggestions from Ross Wightman to integrate:

- I'd try to separate volumes by workload, so keep the 'lots of small files', high churn like environments, code separate from bulk storage like datasets, checkpoints. Possibly even split those too since datasets are largely static and checkpoints are being rotated all the time

- When datasets are on network storage, just like bucket storage, they should consist of large files AND be read as large files (sequentially in large chunks, not mmapped!). Avoid seeking within datasets

- Setups like HF datasets can be deceiving, might look like one big file, but often being mmap'd and the IO read pattern is nuts, like 3-4x more iops than if you'd read them as individual files.
  Mmap loading can be turned off, but if that's the case, for a lot of datasets you move a problem into the DataLoader processes, requiring reading too much data into memory at once. Better awareness of tradeoffs for different use cases, and especially using Iterable streaming when appropriate.

- Note that once your datasets are optimally friendly for a large, distributed network filesystem, they can usually just be streamed from bucket storage in cloud systems that have that option. So better to move them off the network filesystem in that case.

- In a way, bucket storage like s3, via the interface limitations, enforces patterns that are reasonable for storage backends like this. It's ooh, it's mounted as a folder, I can do whatever I want (mmap files, write loads of little ones, delete them all, etc) that's the prob.

- One also cannot expect to treat a distributed filesystem like their local disk. If you separated volumes by workload you'd probably be able to utilize much higher % of the total storage. Don't mix high churn, small files with low churn large files.

- Also, note that once your datasets are optimally friendly for a large, distributed network filesystem, they can usually just be streamed from bucket storage in cloud systems that have that option. So better to move them off the network filesystem in that case.



## Glossary

- NAS: Network Attached Storage
- SAN: Storage Area Network
- DAS: Direct-Attached storage
- NSD: Network Shared Disk


## Which Filesystem to choose

**Parallel FileSystems are the fastest solutions**

- [Lustre FS](https://www.lustre.org/) (Open Source)
- [GPFS](https://en.wikipedia.org/wiki/GPFS) (IBM), recently renamed to IBM Storage Scale, and before that it was called IBM Spectrum Scale.

Both have been around for 2+ decades.

case study: At JeanZay HPC we were saving 2.3TB checkpoint in parallel on 384 processes in 40 secs! This is insanely fast - and it was GPFS over NVME drives.

Other Parallel File Systems I don't yet have experience with:

- [BeeGFS](https://www.beegfs.io)

Most clouds provide at least one implementation of these, but not all. If your cloud provider doesn't provide one and they don't have a fast enough alternative you should reconsider.

**OK'ish solutions**

There are many OK'ish solutions offered by [various cloud providers](#cloud-shared-storage-solutions). Benchmark those seriously before you commit to any. Those are usually quite decent for handling large files and not so much for small files.

case study: As of this writing GCP's Zonal FileStore over NFS solution `python -c "import torch"` takes 20 secs! Once the files are cached it then takes ~2 secs. Installing a conda environment with a handful of prebuilt python packages can take ~30 min!

## FS Clients

You will need to choose which client to use to connect the filesystem to your VM.

The most common choice is: [NFS](https://en.wikipedia.org/wiki/Network_File_System) - has been around for 4 decades. It introduces an additional overhead and slows things down. So if there is a native client supported by your VM, you'd have an overall faster performance using the native client over NFS. For example, GPFS comes with an [NSD](https://www.ibm.com/docs/en/linux-on-systems?topic=configurations-network-shared-disk-nsd) client which is superior to NFS.

## File Block size

If the filesystem you use uses a block size of 16mb, but the average size of your files is 16k, you will be using 1,000 times more disk space than the actual use. For example, you will see 100TB of disk space used when the actual disk space will be just 100MB.

footnote: On Linux the native filesystems typically use a block size of 4k.

So often you might have 2 very different needs and require 2 different partitions optimized for different needs.

1. thousands to millions of tiny files - 4-8k block size
2. few large files - 2-16mb block size

case study: Python is so bad at having tens of thousand of tiny files that if you have many conda environments you are likely to run of inodes in some situations. At JeanZay HPC we had to ask for a special dedicated partition where we would install all conda environments because we kept running out of inodes on normal GPFS partitions. I think the problem is that those GPFS partitions were configured with 16MB block sizes, so this was not a suitable partition for 4KB-large files.

The good news is that modern solutions are starting to introduce a dynamic block size. For example, the most recent GPFS supports sub-blocks. So, for example, it's possible to configure GPFS with a block size of 2mb, with a sub-block of 8k, and then the tiny files get packed together as sub-blocks, thus not wasting too much disc space.



## Cloud shared storage solutions

Here are shared filesystem storage solutions made available by various cloud providers:

- [GCP](https://cloud.google.com/architecture/filers-on-compute-engine)
- [Azure](https://learn.microsoft.com/en-us/azure/virtual-machines/disks-shared)
- [AWS](https://aws.amazon.com/what-is/nas/#seo-faq-pairs#how-can-aws-help-with-storage-solutions)


## Local storage beats cloud storage

While cloud storage is cheaper the whole idea of fetching and processing your training data stream dynamically at training time is very problematic with a huge number of issues around it.

Same goes for dynamic offloading of checkpoints to the cloud.

It's so much better to have enough disc space locally for data loading.

For checkpointing there should be enough local disc space for saving a checkpoint in a fast and reliable way and then having a crontab job or a slurm job to offload it to the cloud. Always keep the last few checkpoints locally for a quick resume, should your job crash, as it'd be very expensive to wait to fetch the checkpoint from the cloud for a resume.

case study: we didn't have a choice and had to use cloud storage for dataloading during IDEFICS-80B training as we had barely any local storage and since it was multimodal data it was many TBs of data. We spent many weeks trying to make this solution robust and it sucked at the end. The biggest issue was that it was very difficult at the time to keep track of RNG state for the DataSampler because the solution we used, well, didn't bother to take care of it. So a lot of data that took a lot of time to create was wasted (not used) and a lot of data was repeated, so we didn't have a single epoch of unique data.


## Beware that you're often being sold only 80% of the storage you pay for

There is a subtle problem with distributed shared storage used on compute nodes. Since most physical disks used to build the large filesystems are only 0.5-2TB large, any of these physical disks can get full before the combined storage gets full. And thus they require constant rebalancing so that there will be no situation where one disk is 99% full and others are only 50% full. Since rebalancing is a costly operation, like most programming languages' garbage collection, it happens infrequently. And so if you run `df` and it reports 90% full, it's very likely that any of the programs can fail at any given time.

From talking to IO engineers, the accepted reality (that for some reason is not being communicated to customers) is that only about 80% of distributed large storage is reliable.

Which means that if you want to have 100TB of reliable cloud storage you actually need to buy 125TB of storage, since 80% of that will be 100TB. So you need to plan to pay 25% more than what you provisioned for your actual needs. I'm not sure why the customer should pay for the technology deficiency but that's how it is.

For example, GCP states that only [89%](https://cloud.google.com/filestore/docs/known-issues#capacity_errors_before_reaching_full_provisioned_capacity) can be used reliably, albeit more than once the storage failed already at 83% for me there. Kudos to Google to even disclosing this as a known issue, albeit not at the point of where a person buys the storage. As in - we recommend you buy 12% more storage than you actually plan to use, since we can only reliably deliver 89% of it.


## Don't forget the checksums

When you sync data to and from the cloud make sure to research whether the tool you use checks the checksums, otherwise you may end up with corrupt during transmission data. Some tools do it automatically, others you have to enable this feature (since it usually comes at additional compute cost and transmission slowdown). Better slow, but safe.

These are typically MD5 and SHA256 checksums. Usually MD5 is sufficient if your environment is safe, but if you want the additional security do SHA256 checksums.



## Benchmarks



### fio

[fio - Flexible I/O tester](https://fio.readthedocs.io/en/latest/) is a commonly used io benchmarking tool, which is quite easy to use.

First install `fio` with `apt install fio` or however your package manager does it.

Here is an example of a read benchmark:

```
base_path=/path/to/partition/
fio --ioengine=libaio --filesize=16k --ramp_time=2s --time_based --runtime=3m --numjobs=16 \
--direct=1 --verify=0 --randrepeat=0 --group_reporting --unlink=1 --directory=$base_path  \
--name=read-test --blocksize=4k --iodepth=64 --readwrite=read
```

Here 16 concurrent read threads will run for 3 minutes. The benchmark uses a block size of 4k (typical for most OSes) with the file size of 16k (a common size of most Python files) in a sequential reading style using non-buffered IO (`O_DIRECT`). So this would be a good benchmark to showing how fast you can import Python modules on 16 concurrent processes.

case study: on one NFS setup we had `python -c "import torch"` taking 20 seconds the first time it was run, which is about 20x slower than the same test on a normal NVME drive. Granted once the files were cached the loading was much faster but it made for a very painful development process since everything was slow.

Important: if you don't use the `--unlink=1` flag make sure to delete `fio`'s work files between different benchmarks - not doing so can lead to seriously wrong reports as `fio` will reuse files it prepared for a different benchmark which must not be re-used if the benchmark parameters have changed. Apparently this reuse is an `fio` feature, but to me it's a bug since I didn't know this nuance and got a whole lot of invalid reports because of it and it took awhile to realize they were wrong.

Going back to the benchmark - the parameters will need to change to fit the type of the IO operation you care to be fast - is it doing a lot of pip installs or writing a checkpoint on 512 processes, or doing a random read from a parquet file - each benchmark will have to be adapted to measure the right thing.

At the beginning I was manually fishing out the bits I was after, so I automated it resulting in [fio-scan](./fio-scan) benchmark that will run a pair of read/write benchmarks on 16KB, 1MB and 1GB file sizes each using a fixed 4k block size (6 benchmarks in total). It uses a helper [fio-json-extract.py](./fio-json-extract.py) to parse the log files and pull out the average latency, bandwidth and iops and report them in a nicely formatted markdown table.

Here is how to run it:
```
git clone https://github.com/stas00/ml-engineering/
cd ml-engineering
cd io

path_to_test=/path/to/partition/to/test
./fio-scan $path_to_test
```
Adapt `path_to_test` to point to the partition path you want to benchmark.

Here is an example of this IO scan on my Samsung SSD 980 PRO 2TB NVME drive ([summary](results/hope-2023-12-20-14-37-02-331702-summary.md)):

* filesize=16k read

| lat msec | bw MBps | IOPS     | jobs |
| -------: | ------: | -------: | ---: |
| 4.0      | 1006.3  | 257614   | 16   |

* filesize=16k write

| lat msec | bw MBps | IOPS     | jobs |
| -------: | ------: | -------: | ---: |
| 3.2      | 1239.1  | 317200   | 16   |

* filesize=1m read

| lat msec | bw MBps | IOPS     | jobs |
| -------: | ------: | -------: | ---: |
| 1.7      | 2400.1  | 614419   | 16   |

* filesize=1m write

| lat msec | bw MBps | IOPS     | jobs |
| -------: | ------: | -------: | ---: |
| 2.1      | 1940.5  | 496765   | 16   |

* filesize=1g read

| lat msec | bw MBps | IOPS     | jobs |
| -------: | ------: | -------: | ---: |
| 1.4      | 2762.0  | 707062   | 16   |

* filesize=1g write

| lat msec | bw MBps | IOPS     | jobs |
| -------: | ------: | -------: | ---: |
| 2.1      | 1943.9  | 497638   | 16   |


As you can see as of this writing this is a pretty fast NVMe drive if you want to use it as a base-line against, say, a network shared filesystem.


### other tools

- [HPC IO Benchmark Repository](https://github.com/hpc/ior)
- [DLIO](https://github.com/argonne-lcf/dlio_benchmark)

XXX: expand on how these are used when I get a chance to try those



### Published benchmarks

Here are some published IO benchmarks:

- [MLPerf via MLCommons](https://mlcommons.org/en/) publishes various hardware benchmarks that measure training, inference, storage and other tasks' performance. For example, here is the most recent as of this writing [storage v0.5](https://mlcommons.org/en/storage-results-05/) results. Though I find the results are very difficult to make sense of - too many columns and no control whatsoever by the user, and each test uses different parameters - so how do you compare things.

Then various benchmarks that you can run yourself:



## Contributors

Ross Wightman
