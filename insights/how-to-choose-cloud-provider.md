# How to Choose a Cloud Provider

Having used multiple compute clouds over long and short terms, and participating in many "discovery" calls, I've learned that it's absolutely crucial to approach the cloud choosing process with an utmost care and dedication. Especially for the long term contracts - you may end up in a 3-year lock-in where you pay millions of dollars and end up having a terrible experience and no way to get out of the contract.

To give you a perspective - a 64-node cluster may easily cost USD$20-50M over a 3 year period. This is often more than what startups pay for the salaries.

I can't stress this enough that choosing a bad 3-year contract may prevent your startup from succeeding.

In this article I'm not going to tell which clouds to avoid, but instead try to empower you to avoid having a bad experience and to have at least a decent one, that will give your company a chance to succeed.

These notes assume you already know what compute you want for your specific workloads. If you don't please skim through the [Accelerator](../compute/accelerator), [Storage](../storage) and [Network](../network) chapters to know what's available out there. Most of the time you want the latest the clouds have to offer.



## Glossary

- CSP: Cloud Service Provider
- SLA: Service-level_agreement
- SLO: Service Level Objective
- TCO: Total Cost of Ownership



## Contracts

If you're paying per hour, you don't need to worry about contracts. But this method isn't good long term because you will be paying many times more and you won't have a steady reliable accelerator foundation. A long term contract at times and with a good negotiator can lead to a 10x in total cost of ownership (TCO) savings (and time)!

### Free Trials

Most cloud service providers (CSPs) have trial programs where you can "kick the tires" for a few days/weeks on a few nodes for free.

Granted, it won't give you an indication of how well the bigger cluster would scale, but it should be sufficient to be able to run quite a few benchmarks and experiments.

It also will give you a good opportunity to check how the provider's customer support works (if any support is included in the free package that is).


### Half-baked solutions

Since a new generation of accelerators happens roughly every 12-18 months and the customer wants those latest accelerators "yesterday" to have a business advantage over their competitors - this gives CSPs barely any time to integrate the new generation of the hardware, test it, adapt their software stack and burn those components in.

So if you want the latest generation as soon as it becomes available you're almost guaranteed to have a bad experience because, well, time is needed to get things right - we are talking about months of waiting. But customers rule - so the CSPs give them what they want, often not quite telling that what the customer gets is not quite ready.

I'm not sure if CSPs are to blame, because often they get the hardware delivery months after it was promised by the manufacturers and, of course, by now they can't keep their promises to the customers, so they just go ahead and deliver...

Then some CSPs develop their own hardware (e.g. network stack) in order to have better margins and then they fail to complete those custom solutions in time, the latest accelerators are there, but the whole system is limping. It's much safer when off-the-shelf components are offered, since those are most likely to be well-tested working components (expect it's likely to cost more).

I think it's OK if the customer wants the hardware early, there should just be an honest disclosure as in: *"look we need some 3 more months to make things solid, if you want the nodes now you can have them but we can't guarantee anything."*

### We-will-do-our-best clause

A lot of the long-term cloud contracts are likely to include a lot of "we will do our best" clauses.

Yet:

1. The customer is not allowed to "do their best" to pay, they are legally obliged to pay the amount they agreed to pay and on time.
2. The customer is not allowed to break a contact before its term runs its course.

In my experience "we will do our best" is demonstrated by Tier-1 clouds by sending 10+ people to the meetings with the customers. Some of them will be clueless and will be just sitting there making the company look resourceful: *"look, we are allocating 10+ people to the problem you're experiencing. You have nothing to worry about"*. Except, most of the time those people can't solve your problem.

What you need is just 2 cloud support people on the call - one product manager and one engineer directly responsible for solving the problem at hand. And in my experience this sort of meeting could take weeks to months to manifest or not at all. Usually one needs to have good connections to be able to escalate the issue to "top brass".

For every critical component of the package you're purchasing you need a quantifiable delivery. For example, if the network you were sold is supposed to run at X GBps at that many nodes doing all-reduce, and you measured it to be significantly lower, there should be a stipulation of what the CSP will do when this happens. How long do they have to fix the problem and whether you can break a contract should this not happen within the agreed by both sides time.

Same goes for storage, accelerators and any other critical component that you plan to rely on.

Of course, it's up to you to negotiate the specific repercussions, but probably the best one is that you stop paying until the problem is fixed. That way there is a huge incentive for the problem to be fixed.

Alas, not paying helps, but not being able to use the compute is still a huge problem. And breaking the contract and migrating to another provider is a huge undertaking not to be taken lightly. But at least there is something you could do if you don't get what you need.

I must also say that it's almost never the problem of the engineers, very often they are amazing experienced people - most of the time it's the issue of management and resource allocation. So please be as gentle as possible with the people you interact with, while firmly demanding a resolution. I know it's a difficult one - more than once I was at the end of the rope, and I couldn't always keep it cool.

### Service Level Agreement

As a continuation of a previous section, a [Service Level Agreement](https://en.wikipedia.org/wiki/Service-level_agreement) (SLA) is an agreement between a service providers and a customer that define various guarantees and expectations with regards to service quality and availability, and various responsibilities.

The other term is Service Level Objective (SLO) where SLA is quantified. For example, an SLO may define a Monthly Uptime Percentage to 99.5%, if the uptime is less than 99.5% the provider credits the customer to a certain percentage of the $$ spent. For example, 10% if the uptime is 99-99.5%, 25% for 95-99%, etc. Here a [GCP SLA](https://cloud.google.com/ai-platform/training-and-prediction/sla?hl=en).

The main category one should care for when renting ML clusters is failing accelerators and/whole nodes. If you paid for 64 nodes but were able to use only 60 you should be reimbursed/credited for those nodes you couldn't use. Your SLA should define the duration of downtime after which the provider starts paying you back and how much.

Same goes for network and storage, albeit those typically fail a lot less often than accelerators, but they do fail.

In general any critical part of the service should have an SLO and clearly defined repercussions if the SLOs aren't met.

Most Tier 1 companies should already include their standard SLAs in the contract. In theory the customer should be able to negotiate those to adapt to their needs, thought it might not always be possible. Sometimes offering to pay more may allow for a better than standard SLO.


### Discuss a contract breaking clause

Both sides should be empowered to experience a mutually beneficial business experience.

Therefore it's critical that you should be able to legally exit the contract should your business experience not be beneficial because the other side is failing to meet the agreed upon expectations.

This, of course, implies not to have a legal battle which can be very costly and Tier-1 clouds have a lot of money to hire the best lawyers, so it might be a losing battle.

It's up to you to negotiate under which circumstances the contract can be cleanly exited before its term runs out.


### Must have paid support included

In one of the companies I worked at our cloud contract didn't include the paid support service and the only support we had was via a customer chat. The paid support was skipped to save costs, but boy did we end up losing days of compute because of that.

Do not try to save here - you will end up losing a lot of money, developer time and hair. Make sure you have a way to submit tickets with priority labels and a defined in the contract expectation to how quickly they will be dealt with.

When you try to use customer chat to solve an urgent problem, there is zero obligation for them to do anything, or at least to do it in a timely manner.

If you're dealing with PMs, you need to know how quickly you could talk directly to the end-point engineer, while removing the middle-man.


### Support during off-hours

Do you get human support for emergencies on weekends/holidays/nights? e.g. On one of the HPCs I used the human support was only available Mon-Fri 9-5.

If this is not available, at the very least ensure that your team can perform cluster resuscitation themselves - and do a drill to ensure this is actually doable. This means you need to have an API to perform all those things without the provider's support.


### Next generation accelerator migration

On average a new generation of accelerators comes out every 12-18 months, but a typical contract is for 3 years. Which means that for about half of that time you will end up using an inferior product.

Nobody wants to use a 2-5x slower accelerator when a much faster version is available, but most customers now are stuck with the old accelerators for the full 3 year duration.

You need to negotiate the ability to move to the new generation before the end of the term, which would obviously require some additional money paid for this to happen.


## Accelerators

This group of questions/issues is specific to accelerators.

### Accelerators need to be burned in

When a new batch of components arrives the provider has to "burn them in" before handing them to customers. This is a process of running an extensive stress testing to detect any accelerators and other system components that are faulty.

If this is not done, the customer ends up discovering the "bad apples" the hard way, while running their workloads. This leads to lost compute and developer time. If the workload uses a few nodes, one failing accelerator isn't a big problem most of the time, but if the workload uses dozens or hundreds of nodes the cost is huge.

It shouldn't be the responsibility of the customer to discover bad accelerators. And while there is no guarantee that the accelerator will not fail after it has been stress tested - it should happen rarely.

Otherwise, a new batch of accelerators often has a 3-10% failure rate, which is huge and very costly to the customer!

So ask your provider how long did they burn in your accelerators/systems for, if at all.

I'm yet to find a golden reference point, but, for example,  [SemiAnalysis](https://semianalysis.com/2024/10/03/ai-neocloud-playbook-and-anatomy/#cluster-deployment-and-acceptance-test) suggests that OEM provider performs a 3-4 weeks burn-in, and then the CSP conducts another 2-3 day long burn-in/acceptance test. So if that's the case you want to ensure that the systems were stress-tested for at least 2-3 days.


### Dealing with accelerator failures

In my experience, while other compute components do fail occasionally, 95% of the time it's the accelerators that fail.

Therefore you need to have a very clear and quick path to an accelerator replacement.

Ideally this process needs to be automated. So you need to ask if there an API to release a broken node and get a replacement. If you have to ask a human to do that, it usually doesn't work too well. The more automated things are, the more efficient the experience.

How many accelerators do you have in the provider-side back up pool available to you? They will usually commit to a certain number of fast replacement per month.

That's said if time is of an essence to your workflows, as most of the time you won't be able to get instant replacements you should always pay for about 10% more nodes than you need. The extra nodes can be used for development and if you have failing nodes during training you can instantly use your own extra nodes.


### Ensure all your nodes are on the same network spine

Unless you're renting 10k gpus, most smaller clusters can easily be co-located on the same network spine - so that it takes the same time to perform inter-node network traffic from any node to any other node.

Ensure that any back up nodes that you're not paying for, but are there to deal with failing accelerators, reside on the same network spine as the nodes you're paying for. If they don't, you are going to have a big problem if you do multi-node training - since that one replacement node will be further away from all other nodes and will slow the ensemble down (the weakest link in the chain).

### Ensure you keep your good accelerators on reboot

You want your cluster to have a fixed allocation. Which means that if you need to re-deploy nodes, and especially if you're planning a downtime, other customers aren't going to grab those nodes!

Once you spent weeks filtering out the bad nodes from the good nodes, it's crucial to keep those nodes to yourself and not start the painful and costly filtering again.

### Do you think you will need to expand?

This is a difficult one, because it's hard to know ahead of time if the amount of nodes you're asking for will need to grow in the future.

Ideally you'd want to discuss this with your provider in case they could plan for your imminent expansion.

Because otherwise, say, you want to double the number of your nodes, but in order to get more nodes, they could only be allocated on another network spine - this is going to be a problem, as it'd impact the training speed.

Chances are that you will have to drop your current allocation and move to another bigger allocation - possibly even in a different region if they don't have local capacity. And moving to a different region can be a very slow and costly experience because you have to move your storage to where your new cluster is. Based on a personal experience - don't treat this lightly.





## Storage

Large and fast storage is very important for both - good developer experience and fast training/finetuning/inference workloads - in particular with regards to loading/saving checkpoints.

### Guaranteed maximum capacity

Ask how much of the storage you will be paying for is guaranteed.

For example, if the Lustre filesystem is used the customer needs to know that they have to over-provision by 25% to get the actual storage capacity they need, because Lustre can fail to write at 80% total storage capacity, because of bad disk balancing design. And the onus of paying for the extra 25% is on the customer!

Most other filesystems I had an experience with typically reach 100% capacity without failing, but it's always good to ask for the specific filesystem you plan to use.

### Know your storage IO requirements

At one of the clouds we used a non-parallel distributed filesystem and the developer experience was absolutely terrible. While dealing with large files was acceptable, the small files experience was extremely slow - it'd take 30 minutes to install a basic Conda environment and 2 minutes to run `python -c "import torch"`. This is because Python has tens of thousands of 4-16kb files and if the file system isn't optimized to handle those and the meta-data servers are weak, it'd be a very frustrating experience.

In general a typical Python shop needs a filesystem that can deal with:
- tens of thousands of tiny files
- few huge files

But, of course, only you know what your workloads' specific requirements are. Also consider the relationship between local storage and remote (shared) storage, as some providers will reduce the size and performance of local drives to save money. In many cases, developers will read data from a shared filesystem that can be cached locally (code libraries, models, datasets). Teaching people how to use [rsync](https://linux.die.net/man/1/rsync) with local NVMe can improve the developer experience, and reduce I/O on the shared filesystem.

Please refer to the notes and guidance in the [Storage chapter](../storage) to know the nuances of storage requirements and their benchmarking.

### What happens when storage fails

With advanced expensive distributed filesystems the chance of failure is relatively small, but it's quite big with cheaper storage solutions.

But it may still happen with any system.

You need to know:
- Who is in charge of fixing the problem?
- How long will it take to recover?
- Who pays for the downtime?
- What are the users to do while there is the problem?

If the resolution will take a long time often one needs to add another temporary filesystem partition to enable people to do their work. And, of course, you will have to pay for it.

### Region migration

A cluster may be forced to migrate to a different region when upgrading to a next generation accelerators or expanding the capacity, if the region you're in doesn't have what you need. The storage has to be in the same region as the accelerators for the workflows to be fast.

The migration event triggers a sometimes very painful storage migration experience.

Here are some critical questions you need to ask long before the migration starts.

- Is the provider responsible for moving your data or is it your responsibility?
- Have you checked that the provided tooling is good enough to move TBs of data in a few hours, or will it takes many days to move? For example, using a storage cloud to migrate will typically drop all file metadata, which can be a huge problem. If you have 5 million tiny files, it could take forever to copy. Unless you use `tar`, but which may take many hours to create and do you have the 2x storage to have 2 copies of your data?
- Are you supposed to pay for the storage and the compute for both overlapping clusters?
- What happens to the files being edited and created while the filesystem is on the move - do you send everybody home while the migration is happening and freeze the filesystem?


### Backup and Archive

Many CSPs only have one tier of file storage available at one price point. However, organiations can have needs for multiple tiers of storage. For example, you might want to archive old model checkpoints or finetuning datasets to cheap, cold storage such as S3 object on HDD.

Having the flexibility to expand your total storage capacity, and keep the "hot" (local NVMe), "warm" (shared NVMe), "cold" (shared HDD), and "archive" (tape) in sync can help improve the resiliency of systems, save money, and allow for easier migration or expansion over time.





## Network

This segment is mostly relevant to those planning to do training and finetuning. If you need to rent accelerators either for inference via large deployments of microservices or for small, on-demand, interactive work (i.e. notebooks) you can safely ignore this information. The only exception is when you plan on inferencing very big models that require more than one node for a single replica.

In general you want to ensure that the offered [intra-node](../network#intra-node-networking) and [inter-node](../network#intra-node-networking) network speeds match the promise and your expectations.

### Ask for the actual performance numbers

Compute theory never matches reality, and the reality may dramatically vary from provider to provider even if they all use the same components, as it depends on the quality of all involved components and how well the racks were designed and put together.

The easiest ask is to request an `all-reduce` benchmark plot over 4-8-16-32-64 nodes (or more if your cluster is more than 64 nodes). You'd expect the bandwidth to gradually become worse with more participating nodes, but not dramatically so. Some networks become very inefficient at higher number of nodes.

Please refer to [Real network throughput](../network#real-network-throughput) for more details.

Ideally you want to benchmark at least a few payloads - the ones that are of a particular interest to you because you know that this is the collective payload you will be using in your workloads. I usually just start by asking for a plot of a big payload of about 4-16GB (16GB would get the best bandwidth on the latest fastest inter-node networks), if the performance drops below 80% of the theoretical GBps, then I know we have a problem.


### Does the network steal from the accelerator memory?

One surprise I experienced on one of the clouds is that when I started using the GPUs I discovered that 5GB of each was already used by the networking software - we managed to reduce it to a lower value, but still we were sold GPUs with less than their memory size and nobody told us about that before we signed the contract.

As accelerators become much bigger this will probably become unimportant, but when you get 75GB of usable memory instead of 80GB on H100 - that's a huge amount of memory lost per GPU.

### Infiniband or Ethernet?

In general, CSPs follow NVIDIA's [DGX SuperPOD Reference Architecture](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/abstract.html) which provides a lot of detail on how to build a rail-optimized InfiniBand network. Rail-optimized basically means that each GPU in an 8-way system connects to it's own leaf switch. Everything else is a standard fat-tree.

However, many of the largest GPU clusters in the world now run RoCEv2 instead of Infiniband. Meta has [proven](https://engineering.fb.com/2024/08/05/data-center-engineering/roce-network-distributed-ai-training-at-scale/) that you can train frontier-class Llama models on a RoCEv2 network. Semianalysis/Fabricated Knowledge show a [significant drop-off](https://www.fabricatedknowledge.com/p/nvidia-waiting-on-blackwell-and-whats?utm_source=post-banner&utm_medium=web&utm_campaign=posts-open-in-app&triedRedirect=true) in NVIDIA's networking attach rate for their GPUs.

Since multi-node training depends on network collectives (i.e. NCCL or RCCL), the type of network can siginificantly impact performance and user experience.


## Security

Though it can sometimes be an afterthought, CSP's approach to security can vary widely. Just achieving a SOC 2 Type 2 compliance certification may not be enough. It is a good idea to check if the machines you'll be using are virtualized. If you're not in a VM, and the cloud provider serves other tenants, you may not trust what they are doing on the machines that you aren't on. It's a good idea to check that your cloud provider is verifying known-good versions of BMC firmware, system and BIOS firmware before provisioning (or re-provisioning) a server for you to use.



## Miscellaneous


### Tier 1 vs Tier 2 clouds

I don't yet have a clear recommendation for whether Tier 1 clouds (AWS, GCP, Azure, etc.) vs emerging smaller Tier 2 clouds are better. My intuition is that Tier 2 clouds are likely to provide a better and more personal support as they have to work harder to secure customers.

Price-wise, Tier 2 clouds in general are cheaper because otherwise they won't be able to compete with Tier 1 clouds. However, it's obvious that their "margin" will be much smaller, because Tier 2 clouds don't have the volume buying power of Tier 1 clouds.

Tier 2 clouds are more likely to be more flexible, have non-mainstream accelerators (e.g., AMD and Intel) and probably are more likely to lend hand at tuning things up at no to little cost.



### Orchestration

A well-oiled node orchestration is critical for successfully using multi-node clusters.

Make sure you know which one you need - usually [SLURM](../orchestration/slurm/), Kubernetes or a combination of the two and make sure it's well supported. Some clouds would only support one of them, or provide a very limited support for another type. These days SLURM is mostly used for training/finetuning and Kubernetes for inference. And there are other [emerging orchestration platforms out there](../orchestration/).

Same as with hardware, depending on whether you're planning to administrate your own cluster you need to know who will deal with any problems. This is a very crucial component of your stack, since if the orchestration is broken, nobody can use the cluster and you lose time/money.


### Up-to-date software/OS versions

Make sure to ask that the provider isn't going to force you into some old versions of the software and an operating system.

I have had experiences where we were forced to use some very old Ubuntu versions because the provider's software stack which we had to use wasn't supporting more recent and up-to-date OS.



### System administration

These days it can be difficult to find a good system administrator that understands the specific needs of the ML workloads, so it's a good idea to ask if some of that work could be offloaded to the CSP. Tier-1 CSPs sub-contract service companies that can provide various degrees of system administration. Smaller clouds are likely to offer their own direct services. They usually have a good grasp of what ML workloads need.

You won't be able to succeed without someone experienced taking care of your cluster. Using your ML engineers to also deal with system administration work can be very counter-productive, since it can be a very time-demanding and interrupting work.

Either hire a system administrator or hire a service company that will do it for you.


## Conclusion

These notes are based on my direct experience and clearly I haven't been exposed to all possible things that may go wrong and wreck havoc with your cluster or make your whole team burn out and lose a lot of their hair. But this should be a good foundation to start thinking about.

Add your own questions, by thinking what's important for you, what failures may prevent you from accomplishing your compute goals.

If you have a particular CSP that you're casing out ask the community about them, especially what pitfalls to avoid with that cloud.

The key message of this article is for you to choose a cloud where your choice hasn't been taken away and that you don't get stuck with a service your developers hate, which is likely to lead to people leaving your company.

If you feel that these notes are overwhelming for you, I occasionally consult helping with due diligence and joining discovery calls. You can contact me at [stas@stason.org](mailto:stas@stason.org?subject=Choosing%20cloud%20consulting).


## Additional reading

- semianalysis.com created a ClusterMax CSP rating system and includes excellent explanations of the different criteria and plans to continue ranking many CSPs. [2025](https://semianalysis.com/2025/03/26/the-gpu-cloud-clustermax-rating-system-how-to-rent-gpus/)
