import time
import torch
from transformers import AutoModelForCausalLM

MODEL = "meta-llama/Llama-3.1-8B"
SEQ = 8192
# MODEL = "NousResearch/Llama-3.2-1B"
# SEQ = 512
BATCH = 1
WARMUP = 3
ITERS = 10
SEED = 42
DTYPE = torch.bfloat16
MODES = {
    "eager": None,
    "default": {},
    "reduce-overhead": {"mode": "reduce-overhead"},
    "max-autotune": {"mode": "max-autotune"},
}
WORKLOADS = ("forward", "forward+backward", "forward+backward+optim")


def main():
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    tokens = torch.randint(0, 30000, (BATCH, SEQ), device=device)
    for workload in WORKLOADS:
        is_training = workload != "forward"
        is_optim = workload == "forward+backward+optim"
        eager = None
        for name, kwargs in MODES.items():
            model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, attn_implementation="sdpa").to(device)
            model.train(is_training)
            if is_training:
                model.config.use_cache = False
                model.gradient_checkpointing_enable()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4) if is_optim else None
            run = model if kwargs is None else torch.compile(model, **kwargs)

            def step():
                if not is_training:
                    with torch.no_grad():
                        run(input_ids=tokens)
                    return
                run.zero_grad(set_to_none=True)
                run(input_ids=tokens, labels=tokens).loss.backward()
                if is_optim:
                    opt.step()

            torch.cuda.synchronize()
            t = time.perf_counter()
            step()
            torch.cuda.synchronize()
            cold = time.perf_counter() - t
            for _ in range(WARMUP):
                step()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(ITERS):
                step()
            torch.cuda.synchronize()
            steady = (time.perf_counter() - t) / ITERS
            if name == "eager":
                eager = steady
            print(f"{workload:22} {name:16} steady={steady * 1000:8.2f}ms cold={cold:7.2f}s speedup={eager / steady:.2f}x")
            del model, run, opt


if __name__ == "__main__":
    main()
