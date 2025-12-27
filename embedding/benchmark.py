import embedding_cuda

import torch.nn as nn
import torch

import time

class EmbeddingCuda(nn.Module):

  def __init__(self , vocab_size : int , embed_dim : int):
    super().__init__()
    self.weight = nn.Parameter(torch.rand(size=(vocab_size , embed_dim)))
    self.embed_dim = embed_dim

  def forward(self , x):
    N = x.shape[1]
    return embedding_cuda.embedding_kernel(x , self.weight , N , self.embed_dim)

def main():
  N = 128
  EMBED_DIM = 4096
  VOCAB_SIZE = 15193

  embed_torch = nn.Embedding(VOCAB_SIZE , EMBED_DIM)
  embed_cuda = EmbeddingCuda(VOCAB_SIZE , EMBED_DIM)

  embedding_weight = torch.rand(size=(VOCAB_SIZE , EMBED_DIM))
  embed_torch.weight = torch.nn.Parameter(embedding_weight.clone())
  embed_cuda.weight = torch.nn.Parameter(embedding_weight.clone())
  input_index = torch.randint(low=0 , high=VOCAB_SIZE ,size=(N,1))

  out = embed_cuda(input_index)
  # print(out)
  
  # print(input_index.shape)
  st = time.monotonic()
  for i in range(10):
    out = embed_torch(input_index)

  et = (time.monotonic() - st) / 10

  
  print(f"Time : {et:.7f}")
  print(f"Throughput : {(N * EMBED_DIM * 4 * 2 + N * 4)/ et * 1e6 }")


  st1 = time.monotonic()

  for i in range(10):
    out1 = embed_cuda(input_index)

  et1 = (time.monotonic() - st1) / 10

  print(f"Time : {et1:.7f}")
  print(f"Throughput : {(N * EMBED_DIM * 4 * 2 + N * 4)/ et1 * 1e6 }")

  print(f"Diff : {torch.max(out - out1)}")

main()

