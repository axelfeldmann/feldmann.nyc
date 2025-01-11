---
layout: post
title: Carry Lookahead Adders
permalink: /blog/carry-lookahead-adder
---
[January 11, 2025]

One of the main things that computers do is add numbers. Because they are computers, these numbers tend to be represented in binary. We can compute the sum of two binary numbers with a base-2 version of the grade-school addition algorithm.

<img src="/posts/carry-lookahead-adder/carry-lookahead-adder-grade-school-addition.drawio.svg" width="350" alt="grade-school-addition">

If we naively synthesize this into a digital circuit, it looks like:

<img src="/posts/carry-lookahead-adder/carry-lookahead-adder-rca.drawio.svg" width="700" alt="ripple-carry-adder">

This design is called a "ripple carry adder" (RCA) and has a very long *critical path* (shown in green). An $n$-bit adder capable of adding 2 $n$-bit numbers has $O(n)$ critical path length. For relevant values of $n$ (e.g. 32, 64), this is terrible! How can we do better?

The answer is called a "carry lookahead adder" (CLA). A CLA has a critical path of $O(\log n)$. Much shorter! In this post, I will share a cool explanation of how CLAs work
that relies primarily on *function composition*. 
I did not come up with this explanation; I learned it from my grad school advisor, [Daniel Sanchez](https://people.csail.mit.edu/sanchez/).

<br/>

---

<br/>

### Observation 1: The hard part is computing carries

_Carries_ are the main thing that make addition hard. Suppose we want to add two $n$-bit numbers `a` and `b`. If we _knew_ all the carries, we could simply compute `sum[i] = a[i] xor b[i] xor carries[i]` and be done with it. xors are cheap circuits and all $n$ bits of `sum` can be computed in parallel.

<img src="/posts/carry-lookahead-adder/carry-lookahead-adder-xors.drawio.svg" width="700" alt="xors">

The problem is that we _don't_ know all $n$ carries. The bigger problem is that if we look at our grade-school adder above, the value of each carry depends on the value of the previous carry-- hence the $O(n)$ critical path.

<br/>

---

<br/>

### Observation 2: Each pair of input bits defines a "carry function"

Let's think about a specific bit index $i$ in the ripple carry adder. We have 3 inputs: `a[i]`, `b[i]`, and `carry_in` (which is index $i-1$'s `carry_out`). `a[i]` and `b[i]` are immediately known. So, really, what we're computing at this index is `carry_in -> carry_out`.  We can call this **carry function** $f_i$. At each index $i$, there is some $f_i$ takes in `carry_in` and computes `carry_out`. 

To actually compute the carry at a particular index, we just need to compose carry functions. For example, if we want to know the the carry at index 3, we just need to compute $f_3(f_2(f_1(f_0(0))))$. We apply $f_0$ to $0$ because the `carry_in` at index $0$ is just $0$.

<img src="/posts/carry-lookahead-adder/carry-lookahead-adder-carry-chain.drawio.svg" width="500" alt="carry-chain">

Importantly, the carry function at each index _is not necessarily the same function_! Each function $f_i$  is determined by the values `a[i]` and `b[i]`. For example, if `a[i] = 0` and `b[i] = 0`, then index $f_i$ will _always_ output a `carry_out` of $0$. If instead `a[i] = 0` and `b[i] = 1`, then index $f_i$ will _propagate_ its `carry_in`.

Concretely:

$$f_i = \begin{cases}
x \mapsto 0 & \text{if } a_i = 0, b_i = 0 \\
x \mapsto x & \text{if } a_i = 1, b_i = 0 \\
x \mapsto x & \text{if } a_i = 0, b_i = 1 \\
x \mapsto 1 & \text{if } a_i = 1, b_i = 1
\end{cases}
$$

<br/>

---

<br/>

### Observation 3: We can get a critical path of $O(\log n)$ function compositions

Let's think about an 8-bit adder. Given our carry functions

$$\begin{bmatrix} f_0 & f_1 & f_2 & f_3 & f_4 & f_5 & f_6 & f_7 \end{bmatrix}$$

What we want is

$$\begin{bmatrix} f_0(0) & (f_1 \circ f_0)(0) & \cdots & (f_7 \circ f_6 \circ f_5 \circ f_4 \circ f_3 \circ f_2 \circ f_1 \circ f_0)(0) \end{bmatrix}$$

For now, let's just assume that function composition is something that we can do efficiently. We'll see why this is true in a bit.

Naively $f_7 \circ f_6 \circ f_5 \circ f_4 \circ f_3 \circ f_2 \circ f_1 \circ f_0$ looks like it has a critical path of 8 function compositions. However, _function composition is associative_. We can instead re-write it as:

$$((f_7 \circ f_6) \circ (f_5 \circ f_4)) \circ ((f_3 \circ f_2) \circ (f_1 \circ f_0))$$


Pictorially, the re-association looks like this:

<img src="/posts/carry-lookahead-adder/carry-lookahead-adder-reassociation.drawio.svg" width="450" alt="reassociation">

After re-associating, our critical path is just _3_ function compositions. Quite an improvement!

This tree composes the function for index 7, but what about all of our other indices? To compute _all_ our indices, we can use a [parallel prefix sum](https://en.wikipedia.org/wiki/Prefix_sum) network-- there are multiple types of prefix sum networks, each with their own tradeoffs, but they all give us $O(\log n)$ critical paths.

Example:

<img src="/posts/carry-lookahead-adder/carry-lookahead-adder-parallel-prefix-sum.drawio.svg" width="450" alt="parallel-prefix-sum">

<br/>

---

<br/>

### Observation 4: Carry function composition *can* be implemented efficiently

So far, we've shown that we can compute carries with $O(\log n)$ carry function composition on the critical path. However, this is only a good idea if we can _efficiently_ compose functions.

The good news is that we can. Each carry function $f_i$ is a function is a **1-bit function**. It takes a single bit as input and produces a single bit as output. The good news is  that there are only 4 possible 1-bit functions:

<img src="/posts/carry-lookahead-adder/carry-lookahead-adder-truth-tables.drawio.svg" width="400" alt="truth-tables">

If we compose two 1-bit functions... we get another 1-bit function! For example, `Generate(Propagate) = Generate`. This means that we can implement function composition as the following 2-bit function[^1][^2]:

<img src="/posts/carry-lookahead-adder/carry-lookahead-adder-composition-table.drawio.svg" width="350" alt="composition-table">

In hardware, this function composition operation can be done with just a few logic gates. As a result, we can easily build a parallel prefix sum network with 1-bit function composition as its operator.

[^1]: for CLAs, we never actually use `Invert`, but I left it in the table for completeness
[^2]: composition is a function that acts on functions! It takes 2 1-bit functions as inputs and produces a 1-bit function as output

<br/>

---

<br/>

### The entire adder

Putting this all together, our carry lookahead adder has 4 main steps:
1. getting the carry functions $f_i$. 
2. composing them using a parallel prefix sum
3. applying the composed carry functions to $0$
4. xoring the carries with `a` and `b` to actually do the addition

In Python-like pseudocode, this might look something like this:
```python
def get_func(ai, bi):
    match (ai, bi):
        case (0, 0): return Kill
        case (0, 1): return Propagate
        case (1, 0): return Propagate
        case (1, 1): return Generate

def adder(a, b):
    fs = [ get_func(ai, bi) for (ai, bi) in zip(a, b) ]
    composed_fs = prefix_sum(fs, compose)
    carry_outs = [ cf(0) for cf in composed_fs ]

    # shift the carries to align carry_outs with the next index
    shifted_carry_outs = [ 0 ] + carry_outs[:-1]
    return a xor b xor shifted_carry_outs
```

And... that's it.

The coolest part here is that this approach generalizes beyond adders. The only operation we did here was function
composition... nothing else!
If we can express *any* computation as a chain of functions that can be *efficiently* composed, we can apply the same
trick to get an $O(\log n)$ critical path. This is really powerful! 

Notes:
- if we represent our carry functions as Python `lambda`s, we don't actually get the efficient function composition discussed in Observation 4, but I think it makes it clearer what's going on
- there are many different `prefix_sum` networks, with their own tradeoffs. Generating good prefix sum networks is an [active research area](https://www.sci.utah.edu/publications/Roy2021a/PrefixRL_Optimization_of_Parallel_Prefix_Circuits_using_Deep_Reinforcement_Learning.pdf)

<br/>

---

<br/>