# batch_size = 1
# block_size = 1024

# input_ids = torch.randint(50000, (batch_size, block_size), device='cuda') #torch.randint(50304, (batch_size, block_size), device=device)
# target_ids = torch.randint(50000, (batch_size, block_size), device='cuda')

# model.eval()
# model_hf.eval()

# N_ITERS = 10

# def timed(fn):
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
#     start.record()
#     result = fn()
#     end.record()
#     torch.cuda.synchronize()
#     return result, start.elapsed_time(end) / 1000

# def evaluate(mod):
#     with ctx:
#         return mod(input_ids, target_ids)

# def evaluate_hf(mod):
#     with ctx:
#         return mod(input_ids, labels=target_ids)
    
# # evaluate_opt = torch.compile(evaluate)
# # torch._dynamo.config.verbose = True


# eager_times = []
# compile_times = []

# model_hf.cuda()
# for i in range(N_ITERS):
# #     _, eager_time = timed(lambda: evaluate(model))
#     _, eager_time = timed(lambda: evaluate_hf(model_hf))
#     eager_times.append(eager_time)
#     print(f"eager eval time {i}: {eager_time}")
# del _
# print("~" * 10)
# model_hf.cpu()
# torch.cuda.empty_cache()

# model.cuda()
# compile_times = []
# for i in range(N_ITERS):
#     _, compile_time = timed(lambda: evaluate(model))
#     compile_times.append(compile_time)
#     print(f"compile eval time {i}: {compile_time}")
    
# del _
# print("~" * 10)
# model.cpu()
# torch.cuda.empty_cache()

# import numpy as np
# eager_med = np.median(eager_times)
# compile_med = np.median(compile_times)
# speedup = eager_med / compile_med
# print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
# print("~" * 10)