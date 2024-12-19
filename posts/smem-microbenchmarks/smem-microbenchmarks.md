---
layout: post
title: Notes About CUDA Shared Memory Banks
permalink: /blog/smem-microbenchmarks
---

Writing performant CUDA kernels requires a lot of thinking about memory. Loading/storing data from/to global memory is _much_ more throughput-constrained than doing arithmetic. To mitigate this, each of the GPU's streaming multiprocessors has its own small scratchpad memory where it can stash frequently used data and access it more cheaply. This scratchpad can be managed either _implicitly_ as an L1 cache or _explicitly_ as "shared memory."

On the whole, shared memory is great! When used correctly, it allows us to write fast kernels. However, it also has sharp edges-- using it incorrectly and incurring _bank conflicts_ can severely limit its effectiveness.

In this writeup, I will attempt to uncover/explain some of the performance characteristics of CUDA shared memory through microbenchmarking. I will use an Nvidia A100 GPU for all experiments.

<br/>

---

<br/>

### Warps and Banks


CUDA threads are organized into **warps**. Each warp is a group of 32 threads that execute _in lockstep_-- they execute the same instruction at the "same time" [^1].  The position of a thread within a warp is called its **lane**.

Shared memory is organized into 32 **banks**. Each bank is a slice of SRAM that can load or store 4 bytes of data every cycle.

The fact that both of these numbers are 32 _is not a coincidence_. The system is designed such that if every lane in a warp loads data from "its own" bank, then all 32 loads can be completed in a single cycle.

Shared memory addresses are mapped to banks via the function `bank(addr) := (addr / 4) % 32`. This means that if we have a shared memory array of  floats
```cuda
__shared__ float s[64];
```
then its elements be round-robin distributed across banks as follows:

![array.drawio.svg](/posts/smem-microbenchmarks/array.drawio.svg)

<br/>

---

<br/>

### Conflict-Free Accesses

As a baseline, we'll first need to develop a microbenchmark where each lane loads from a different bank of shared memory:

![no-conflicts.svg](/posts/smem-microbenchmarks/no-conflicts.svg)

Code[^2]:
```cuda
__global__ void conflict_free() {
    __shared__ float s[8][32];
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;

    float* ptr = &s[warp_id][lane_id];
    int addr = (int)ptr & 0xFFFF;

    for (int j = 0; j < num_iters; j++) {
        asm volatile ("ld.volatile.shared.f32 %0, [%1];"
                        : "=f"(r1)
                        : "r"(addr));
    }
}
// launched with conflict_free<1,dim3(32, 8)>();
```

This microbenchmark features a single thread block with 8 warps. Inside each warp, each lane is issuing nice, aligned loads to its own shared memory bank. This should be "speed of light" shared memory usage. 

On an A100, this kernel with `num_iter = 100'000` takes ~0.57 milliseconds. We have 8 warps, each issuing 100,000 loads to each bank, so 800,000 loads/bank. (800,000 loads/bank) / (0.57 ms)  = 1.4 gigaloads/bank/second. This is equal to the A100's peak frequency of 1.4GHz. So, in essence, we're getting the 1 load/bank/cycle that we expect.

<br/>

---

<br/>

### Bank Conflicts

OK, but what if all 32 lanes try to load different values from the same bank? Diagram:
<img src="/posts/smem-microbenchmarks/all-conflicts.svg" width="500" alt="all-conflicts">

Code:
```cuda
__global__ void all_conflicts() {
    __shared__ float s[8][32];
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;

    float* ptr = &s[lane_id][0];
    int addr = (int)ptr & 0xFFFF;

    for (int j = 0; j < num_iters; j++) {
        asm volatile ("ld.volatile.shared.f32 %0, [%1];"
                        : "=f"(r1)
                        : "r"(addr));
    }
}
// launched with conflict_free<1,dim3(32, 8)>();
```

Running this code (also with `num_iters = 100'000`) takes 18.2 ms-- almost *exactly* 32 times longer than the conflict-free example! This is because within each warp, all 32 lanes are trying to load different values from Bank 0, and these loads must therefore be serialized.

In this example, _all_ of the lanes in _all_ of our warps load exclusively from Bank 0. Bank 0 can only serve one load per cycle, so of course it will be 32x slower than the conflict-free version. However, what if _different warps_ have bank conflicts on _different banks_? So, what if Warp 0's lanes all try to load from Bank 0 and Warp 1's lanes all try to load from Bank 1?

Diagram:
![warp-conflict.svg](/posts/smem-microbenchmarks/warp-conflict.svg)

In the code, this just requires changing: `float* ptr = &s[lane_id][0];` to `float* ptr = &s[lane_id][warp_id];`

It turns out that this also takes 18.2 ms per iteration. Even though each warp is bank-conflicting on a _different_ bank, it seems like the shared memory system is unable to overlap accesses from different warps. A bit sad, but it makes sense. The alternative would be significantly more complicated and costly to implement in hardware.

<br/>

---

<br/>

### Multicasts

What if all of a warp's lanes load the _same_ address from one bank?
![broadcast.svg](/posts/smem-microbenchmarks/broadcast.svg)

In our microbenchmark code, this requires setting `ptr` to `&s[warp_id][0];`

Running this on the A100 with `num_iters = 100'000` takes 0.57 ms, the same amount of time as with conflict-free loads. This is because the hardware only loads a single address in Bank 0 then _broadcasts_ the loaded value to all 32 lanes. The scratchpad is performing fewer loads (only 1 bank is actually doing anything), but each thread is still able to execute 1 load/cycle.

_Every_ lane in a warp loading the same value is easy mode. What if instead, we pick a few "random" groups of lanes in each warp and have each group load the same value from a given bank?

We can achieve this by using a simple hash function to set `ptr`:
```
unsigned int hash = (lane_id * 2654435761u) >> 16;
volatile float* ptr = &sh[warp_id][hash % 32];
```
This results in the following rat's nest of a load pattern:
![ratnest.svg](/posts/smem-microbenchmarks/ratnest.svg)

With this pattern, some of the banks (such as Bank 1) get _no_ loads, while others (such as Bank 0) get several. Note that there are no bank conflicts here! Just several values that need to be _multicast_ to a handful of threads. Indeed, some values, such as `s[15]` are only loaded by a single thread.

Impressively, executing this benchmark _also_ takes 0.57ms for 100,000 loads/thread. The GPU's shared memory system is able to do _arbitrary_ multicasts at the same speed as it's able to do conflict-free loads and full-warp broadcasts! Pretty cool.

<br/>

---

<br/>

### Wide Vectorized Loads

So far, all of our microbenchmarks have dealt exclusively with 32-bit loads. These loads map well onto the shared memory banking scheme because every bank is also 32 bits wide. However, Nvidia GPUs also support wider "vector" load types. These load instructions, such as `ld.shared.v4` load _multiple_ contiguous 32-bit values from shared memory into registers.

Suppose we want to use 4-wide vector loads to load 128 contiguous floats into registers. Under this scheme, lane `i` will load `s[4*i]`, `s[4*i+1]`, `s[4*i+2]`, and `s[4*i+3]` all in a single instruction. 

The code looks like this:
{% raw %}
```cuda
__global__ void vectorized_loads() {
    __shared__ float sh[8][128];
    
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
        
    float4* ptr = reinterpret_cast<float4*>(&sh[warp_id][lane_id * 4]);
    int addr = (int)ptr & 0xFFFF;

    float4 r;
    for (int j = 0; j < num_iters; j++) {
        asm volatile ("ld.volatile.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
                        : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
                        : "r"(addr));
    }
}
```
{% endraw %}

Diagram (only first 16 out of 32 lanes shown):
![v4loads.svg](/posts/smem-microbenchmarks/v4loads.svg)

Immediately, we see a possible problem: both Lane 0 and Lane 8 are loading from the same set of 4 banks. What's worse is that Lane 16 and Lane 24 (not shown in the diagram) will _also_ be loading from this same set of 4 banks! We saw earlier that bank conflicts are terrible for performance, so it seems like we're headed for disaster.

Running this code with `num_iter = 100'000` takes 2.27ms, 4x more time than the conflict-free loads. However, _it is loading 4x more data per iteration_ (512B vs. 128B). So, it's still achieving 4-bytes load/lane/cycle. 

It can achieve this "speed of light" throughput despite the possible bank conflicts by _careful hardware load scheduling_. Each lane is going to need to load from 4 different banks. However, each lane does not need to load from its 4 banks _in the same order_. So, for example: Lane 0 could load `s[0]` then `s[1]` then `s[2]` then `s[3]`on 4 consecutive cycles.  But, Lane 8 could load `s[33]` then `s[34]` then `s[35]`, then `s[32]` on those same 4 consecutive cycles. By interleaving the accesses to different banks, each warp is able to load 512B in 4 cycles _with no bank conflicts_. 

Note: _we could_ do this in software too!
```cuda
int start = (lane_id / 8) % 4;
for (int j = 0; j < num_iters; j++) {
    for (int i = 0; i < 4; i++) {
        int offset = (i + start) % 4;
        asm volatile ("ld.volatile.shared.f32 %0, [%1];"
                      : "=f"(r)
                      : "r"(addr + offset * 4));
    }
}
```
This code forces each group of 8 lanes to execute the 4 loads in a cyclically-shifted order. It achieves the same performance as the `ld.shared.v4` microbenchmark. However, this is clearly much more annoying to write, so we may as well use the vector loads :)

<br/>

---

<br/>

### Shared Vector Loads

We can also use vector loads to _amplify_ shared memory bandwidth. Specifically, each pair of adjacent lanes can load _the same_ 2-vector of floats from shared memory.

Diagram:
![2amplify.svg](/posts/smem-microbenchmarks/2amplify.svg)

Code:
{% raw %}
```cuda
__global__ void yifan_multicast(float* result) {
    __shared__ float sh[8][32];

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;

    float* ptr = &sh[warp_id][(lane_id / 2) * 2];z
    int addr = (int)ptr & 0xFFFF;

    float2 r;
    for (int j = 0; j < num_iters; j++) {
        asm volatile ("ld.volatile.shared.v2.f32 {%0,%1}, [%2];"
                    : "=f"(r.x), "=f"(r.y)
                    : "r"(addr));
    }
}
```
{% endraw %}

This code, like the conflict-free loads, also runs in 0.57 ms for `num_iter = 100'000`. If we look from the perspective of the shared memory banks, each bank is servicing 1 load/cycle. This is the same as the conflict-free loads example. However, if we look from the perspective of the lanes, each lane is getting _2_ loads/cycle worth of data-- both `r.x` and `r.y`. Every value is being multicast to _both_ lanes that load it without introducing any performance penalties. This is very powerful! Despite the actual per-bank throughput is unchanged, our threads are able to load _more_ stuff by taking advantage of this vector multicast.

We can try the same thing with 4-wide vector loads:

{% raw %}
```cuda
__global__ void yifan_multicast(float* result) {
    __shared__ float sh[8][32];

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;

    float* ptr = &sh[warp_id][(lane_id / 4) * 4];z
    int addr = (int)ptr & 0xFFFF;

    float4 r;
    for (int j = 0; j < num_iters; j++) {
        asm volatile ("ld.volatile.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
                        : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w)
                        : "r"(addr));
    }
}
```
{% endraw %}

However, in this case, we observe a 2x slowdown (~1.14ms). Because every instruction is loading 4 values,
we're still getting 2-load/thread/cycle throughput (like the 2-wide vector load example), but we're *not* getting
any additional throughput improvement over the 2-wide vector loads.
I don't really know why this is the case, but maybe it's a lack of register file write ports?
Definitely something to look into further.


<br/>

---

<br/>

### Conclusion

In this writeup, we looked at several different shared memory behaviors and tested them using some simple microbenchmarks. 

Thanks to my good friend [Yifan Yang](https://yang-yifan.github.io) who helped with inventing + debugging microbenchmarks.

<br/>

---

<br/>

[^1]: [Not quite](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/#update_legacy_warp-level_programming), but close enough.
[^2]: One interesting thing that I learned while putting together these microbenchmarks is that you can mark ptx instructions as volatile. Weird! What does it even mean for an instruction to be volatile? It turns out that  marking ptx instructions as volatile is a directive for the ptx assembler that lowers them to SASS. This volatile marking stops the ptx assembler from optimizing out all of my loads.
