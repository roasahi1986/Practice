// Copyright (C) 2023 lusyu1986@icloud.com

#ifndef COMPUTATIONALADVERTISING_SRC_EMBEDDING_EMBEDDING_H_
#define COMPUTATIONALADVERTISING_SRC_EMBEDDING_EMBEDDING_H_

#include <vector>

namespace computational_advertising {

class Embedding {
 public:
  Embedding() noexcept(false) {}
  ~Embedding() {}

  Embedding& operator=(const Embedding&) = delete;
  Embedding(const Embedding&) = delete;

  virtual void get_embedding(const std::vector<uint64_t>& ids, std::vector<float*> *embeddings) noexcept(false) = 0;
};

class LocalEmbedding : public Embedding {
 public:
  LocalEmbedding() noexcept(false) {}
  ~LocalEmbedding() {}

  LocalEmbedding& operator=(const LocalEmbedding&) = delete;
  LocalEmbedding(const LocalEmbedding&) = delete;

  void get_embedding(const std::vector<uint64_t>& ids, std::vector<float*> *embeddings) noexcept(false) override {}
};

class RemoteEmebedding : public Embedding {
 public:
  RemoteEmebedding() noexcept(false) {}
  ~RemoteEmebedding() {}

  RemoteEmebedding& operator=(const RemoteEmebedding&) = delete;
  RemoteEmebedding(const RemoteEmebedding&) = delete;

  void get_embedding(const std::vector<uint64_t>& ids, std::vector<float*> *embeddings) noexcept(false) override {}
};

}  // namespace computational_advertising

#endif  // COMPUTATIONALADVERTISING_SRC_EMBEDDING_EMBEDDING_H_
