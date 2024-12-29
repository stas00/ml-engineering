# How to Choose an Accelerator Cloud

Having used multiple accelerator clouds over long and short terms, and participating in many "discovery" calls, I've learned that it's absolutely crucial to approach the cloud choosing process with an utmost care and dedication. Especially for the long term contracts - you may end up in a 3-year lock-in where you pay millions of dollars and end up having a terrible experience and no way to get out of the contract.

To give you a perspective - a 64-node cluster may easily cost $20-50M over a 3 year period. This is often more than what startups pay for the salaries.

I can't stress this enough that choosing a bad 3-year contract may prevent your startup from succeeding.

In this article I'm not going to tell which clouds to avoid, but instead try to empower you to avoid having a bad experience and to have at least a decent one, that will give your company a chance to succeed.

These notes assume you already know what compute you want for your specific workloads. If you don't please skim through the [Accelerator](../compute/accelerator), [Storage](../storage) and [Network](../network) chapters to know what's available out there. Most of the time you want the latest the clouds have to offer.


## Contracts

If you're paying per hour, you don't need to worry about contracts. But this method isn't good long term because you will be paying many times more and you won't have a steady reliable accelerator foundation. A long term contract at times and with a good negotiator can lead to a 10x in TCO savings (and time)!

### Half-baked solutions

Since a new generation of accelerators happens roughly every 12-18 months and the client wants those latest accelerators yesterday to have a business advantage over their competitors - this gives cloud providers barely any time to integrate the new generation of the hardware, test it, adapt their software stack and bake those components in.

So if you want the latest generation as soon as it becomes available you're almost guaranteed to have a bad experience because, well, time is needed to get things right - we are talking about months of waiting. But customers rule - so the cloud providers give them what they want, often not quite telling that what the customer gets is not quite ready.

I'm not sure if cloud providers are to blame, because often they get the hardware delivery months after it was promised by the manufacturers and, of course, by now they can't keep their promises to the customers, so they just go ahead and deliver...

Then some cloud providers develop their own hardware (e.g. network stack) in order to have better margins and then they fail to complete those custom solutions in time, the latest accelerators are there, but the whole system is limping. It's much safer when an off-the-shelf components are offered, since those are most likely to be well-tested working components (expect it's likely to cost more).

I think it's OK if the customer wants the hardware early, there should just be an honest disclosure as in: *"look we need some 3 more months to make things solid, if you want the nodes now you can have them but we can't guarantee anything."*

### We-will-do-our-best clause

A lot of the long-term cloud contracts are likely to include a lot of "we will do our best" clauses.

Yet:

1. The client is not allowed to "do their best" to pay, they are legally obliged to pay the amount they agreed to pay.
2. The client is not allowed to break a contact before its term runs its course.

In my experience "we will do our best" is performed by Tier-1 clouds by sending 10+ people to the meetings with the customers. Most of them will be clueless and will be just sitting there making the company look resourceful: *"look, we are allocating 10+ people to the problem you're experiencing. You have nothing to worry about"*. Except, most of the time those people can't solve your problem.

What you need is just 2 cloud support people on the call - one product manager and one engineer directly responsible for solving the problem at hand. And in my experience this could take weeks to months to happen or not at all. Usually one needs to have good connections to be able to escalate the issue to "top brass".

For every critical component of the package you're purchasing you need a quantifiable delivery. For example, if the network you were sold is supposed to run at X GBps at that many nodes doing all-reduce, and you measured it to be significantly lower, there should be a stipulation of what the cloud will do when this happens. How long do they have to fix the problem and whether you can break a contract should this not happen within the agreed by both sides time.

Same goes for storage, accelerators and any other critical component that you plan to rely on.

Of course, it's up to you to negotiate the specific repercussions, but probably the best one is that you stop paying until the problem is fixed. That way there is a huge incentive for the problem to be fixed.

I must also say that it's almost never the problem of the engineers, very often they are amazing experienced people - most of the time it's the issue of management and resource allocation. So please be as gentle as possible with the people you interact with, while firmly demanding a resolution. I know it's a difficult one - more than once I was at the end of the rope, and I couldn't always keep it cool.


### Discuss a contract breaking clause

Both sides should be empowered to experience a mutually beneficial business experience.

Therefore it's critical that you should be able to legally exit the contract should your business experience not be beneficial because the other side is failing to meet the agreed upon milestones.

This, of course, implies not to have a legal battle which can be very costly and Tier-1 clouds have a lot of money to hire the best lawyers, so it might be a losing battle.

It's up to you to negotiate under which circumstances the contract can be cleanly exited before its term runs out.


### Must have paid support included

In one of the companies I worked at our contract didn't include the paid support service and the only support we had was via customer chat. The paid support was skipped to save costs, but boy did we end up losing days of compute because of that.

Do not try to save here - you will end up losing a lot of money, developer time and hair. Make sure you have a way to submit tickets with priority labels and a defined in the contract expectation to how quickly they will be dealt with.

When you try to use customer chat, there is zero obligation for them to do anything, or at least to do it in a timely manner when things are urgent.

If you're dealing with PMs, you need to know how quickly you can ask to get the end-point engineer
involved, while removing the middle-man.


### Support during off-hours

Do you get human support for emergencies on weekends/holidays/nights? e.g. On one of the HPCs I used the human support was only available Mon-Fri 9-5.

If this is not available, at the very least ensure that your team can perform cluster resuscitation themselves - and do a drill to ensure this is actually doable. This means you need to have an API to perform all those things without the provider's support.


### Next generation accelerator migration

On average a new generation of accelerators comes out every 12-18 months, but a typical contract is for 3 years. Which means that for about half of that time you will end up using an inferior product.

Nobody wants to use a 2-5x slower accelerator when a much faster version is available, but most customers now are stuck with the old accelerators for the full 3 year duration.

You need to negotiate the ability to move to the new generation before the end of the term, which would obviously require some additional money paid for this to happen.





## Accelerators

This group of questions/issues is specific to accelerators

### Accelerators need to be burned in

When a new batch of components arrives the provider has to "burn them in" before handing them to customers. This is a process of running an extensive stress testing to detect any accelerators and other system components that are faulty.

If this is not done, the customer ends up discovering the "bad apples" the hard way, while running their workloads. This leads to lost compute and developer time. If the workload uses a few nodes, one failing accelerator isn't a big problem most of the time, but the workload uses dozens or hundreds of nodes the cost is huge.

It shouldn't be a responsibility of the customer to discover bad accelerators. And while there is no guarantee that the accelerator will still fail after it has been stress tested - it should happen rarely.

Otherwise a new batch of accelerators often has a 3-10% failure rate, which is huge and very costly to the customer!

So ask your provider how long did they burn in your accelerators/systems for if at all.

I'm yet to find a golden reference point, but, for example,  [SemiAnalysis](https://semianalysis.com/2024/10/03/ai-neocloud-playbook-and-anatomy/#cluster-deployment-and-acceptance-test) suggests that OEM provider performs a 3-4 weeks burn-in, and then the actual cloud provider conducts another 2-3 day burn-in/acceptance test. So if that's the case you want to ensure that the systems were stress-tested for at least 2-3 days.


### Dealing with accelerator failures

In my experience, while other compute components do fail occasionally, 95% of the time it's the accelerators that fail.

Therefore you need to have a very clear and quick path to an accelerator replacement.

Ideally this process needs to be automated. So you need to ask if there an API to release a broken node and get a replacement. If you have to ask a human to do that that usually doesn't work too well. The more automated things are the more efficient the experience.

How many accelerators do you have in the provider-side back up pool? They will usually commit to a certain number of replacement per months.


### Ensure all your nodes are on the same network spine

Unless you're renting 10k gpus, most smaller clusters can easily be co-located on the same network spine - so that it takes the same time to perform inter-node network traffic.

Ensure that any back up nodes that you're not paying for, but are there to deal with failing accelerators, reside on the same network spine as the nodes you're paying for. If they don't, you are going to have a huge problem if you do multi-node training - since that one replacement node will be further away from all other nodes and will slow everything down.

### Ensure you keep your good accelerators on reboot

You want your cluster to have a fixed allocation. Which means that if you need to re-deploy nodes, and especially if you're planning a downtime, other customers aren't going to grab those nodes!

Once you spent weeks filtering out the bad nodes from the good nodes, it's crucial to keep those nodes to yourself and not start the wasteful filtering again.

### Do you think you will need to expand?

This is a difficult one, because it's hard to know ahead of time if the amount of nodes you're asking for will need to grow in the future.

Ideally you'd want to discuss this with your provider in case they could plan for your imminent expansion.

Because otherwise, say you want to double the number of your nodes, but in order to get more nodes, they could only be allocated on another network spine - this is going to be a problem as it'd impact the training speed.

Chances are that you will have to drop your current allocation and move to another bigger allocation - possibly even in a different region if they don't have local capacity. And moving to a different region can be a very slow and costly experience because you have to move your storage to where your new cluster is. Based on personal experience - don't treat this lightly.





## Storage

Large and fast storage is very important for both - good developer experience and fast training/inference workloads - in particular wrt loading/saving checkpoints.

### Guaranteed maximum capacity

Ask how much of the storage you will be paying for is guaranteed.

For example, if the Lustre filesystem is used the client needs to know that they have to over-provision by 25% to get the actual storage capacity they need, because Lustre can fail to write at 80% total storage capacity, because of bad disk balancing design. And the onus of paying extra 25% is on customer!

Most other filesystems I had an experience typically reach 100% capacity without failing, but it's always good to ask for the specific filesystem you plan to use.

### Know your storage IO requirements

At one of the clouds we used a non-parallel distributed filesystem and the developer experience was absolutely terrible. While dealing with large files was good, the small files experience was extremely slow - it'd take 30min to install a basic conda environment and 2 minutes to run `python -c "import torch"`. This is because Python has tens of thousands of tiny files and if the file system isn't optimized to handle those and the meta-date servers are weak, it'd be a very frustrating experience.

In general a typical Python shop needs a filesystem that can deal with:
- many tiny files
- few huge files

But, of course, only you know what your workloads' specific requirements are.

Please refer to the notes and guidance in the [../storage/](Storage chapter) to know the nuances of storage requirements and their benchmarking.

### What happens when storage fails

With advanced expensive distributed filesystems the chance of failure is relatively small, but it's quite big with cheaper storage solutions.

But it may still happen with any system.

You need to know:
- Who is in charge of fixing the problem
- How long will it take to recover
- Who pays for the downtime

### Region migration

A cluster may be forced to migrate to a different region when upgrading to a next generation accelerators or expanding the capacity, if the region you're in doesn't have what you need. The storage has to be in the same region as the accelerators to be fast.

The migration event triggers a sometimes very painful storage migration experience.

Here are some critical questions you need to ask long before the migration starts.

- Is the provider responsible for moving your data or is it your responsibility?
- Have you checked that the provided tooling is good enough to move TBs of data in a few hours, or will it takes many days to move? For example, using a storage cloud to migrate will typically drop all file metadata, which can be a huge problem. If you have 5 million tiny files, it could take forever to copy. Unless you use `tar`, but which may take many hours to create and do you have the 2x storage to have 2 copies of your data?
- Are you supposed to pay for the storage and the compute for both overlapping clusters?
- What happens to the files being edited and created while the filesystem is on the move - do you send everybody home while the migration is happening and freeze the filesystem?




## Network

In general you want to ensure that the offered [intra-node](../network#intra-node-networking) and [inter-node](../network#intra-node-networking) network speed matches your expectations.

### Ask for the actual performance numbers

Theory never matches reality, and the reality may dramatically vary from provider to provider even if they all use the same components, as it depends on the quality of all involved components and how well the racks were designed and put together.

The easiest ask is to show an all-reduce benchmark plot over 4-8-16-32-64 nodes. You'd expect the bandwidth to gradually become worse with more participating nodes, but not dramatically so. Some networks become very inefficient at higher number of nodes.

Please refer to [Real network throughput](../network#real-network-throughput) for more details.

Ideally you want to benchmark at least a few payloads - the ones that are of a particular interest to you because you know that this is the collective payload you will be using in your workloads. I usually just start by asking for a plot of a big payload of about 4-16GB (16GB would get the best bandwidth on the latest fastest inter-node networks), if the performance drops below 80% of the theoretical GBps, I know we have a problem.


### Does the network steal from the accelerator memory?

One surprise I experienced on one of the clouds is that when I started using the GPUs I discovered that 5GB of each was already used by the networking software - we managed to reduce it to a lower value, but still we were sold GPUs with less than their memory size and nobody told us about that before we signed the contract.

As accelerators become much bigger this will probably become unimportant, but when you get 75GB usable instead of 80GB - that's a huge amount of memory lost per GPU.



## General

### Orchestration

A well-oiled node orchestration is critical for successfully using multi-node clusters.

Make sure you know which one you need - usually [SLURM](../orchestration/slurm/), Kubernetes or a combination of the two and make sure it's well supported. Some clouds would only support one of them, or provide a very limited support for another type.

Same as with hardware, depending on whether you're planning to administrate your own cluster you need to know who will deal with any problems. This is a very crucial component since if the orchestration is broken, nobody can use the cluster and you lose time/money.


### Up-to-date software/OS versions

Make sure to ask that the provider isn't going to force you into some old versions of the software and an operating system.

I have had experiences where we were forced to use some very old Ubuntu versions because the provider's software stack which we had to use wasn't supporting more recent and up-to-date software stack.



### System administration

These days it can be difficult to find a good system administrator that understands the specific needs of the ML workloads, so it's a good idea to ask if some of that work could be offloaded to the cloud provider. Tier-1 cloud providers sub-contract service companies that can provide various degrees of system administration. Smaller clouds are likely to offer their own direct services. They usually have a good grasp of what ML workloads need.

You won't be able to succeed without someone experienced taking care of your cluster. Using your ML engineers to also deal with sysadmin work can be very counter-productive since it can be a very time-demanding task.

Either hire a system administrator or hire a service company that will do it for you.


## Conclusion

These notes are based on my direct experience and clearly I haven't been exposed to all possible things that may go wrong and wreck havoc with your cluster or make your whole team burn out and lose a lot of their hair. But this should be a good foundation to start thinking about.

Add your own questions, by thinking what's important for you, what failures may prevent you from accomplishing your compute goals.

If you have a particular cloud provider that you're casing out ask the community about them and what particular pitfalls to avoid with that cloud.

The key message of this article is for you to choose a cloud where your choice hasn't been taken away and that you don't get stuck with a service your developers hates, which is likely to lead to people leaving your company.

If you feel that these notes are overwhelming for you, I occasionally consult helping with due diligence and joining your discovery calls. You can contact me at [stas@stason.org](mailto:stas@stason.org?subject=Choosing%20cloud%20consulting).
