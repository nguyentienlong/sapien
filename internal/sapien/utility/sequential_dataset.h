// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#ifndef INTERNAL_SAPIEN_UTILITY_SEQUENTIAL_DATASET_H_
#define INTERNAL_SAPIEN_UTILITY_SEQUENTIAL_DATASET_H_

#include <stddef.h>
#include <vector>

namespace sapien {
namespace internal {

template<typename T>
class SequentialDataset {
 public:
  SequentialDataset(const size_t m, const size_t n, const T* matrix,
                    const T* targets, const T* weights = NULL);
  ~SequentialDataset();

  // We explicitly delete default constructor, copy constructor, and
  // assignment operator
  SequentialDataset() = delete;
  SequentialDataset(const SequentialDataset&) = delete;
  SequentialDataset& operator=(const SequentialDataset&) = delete;

  const size_t n_features;
  const size_t n_samples;

  struct Sample {
    const T* x;
    T target;
    T weight;

    Sample(const T* x, const T target, const T weight);
  };

  const Sample operator[](const size_t i) const;
  const Sample at(const size_t i) const {
    return this->operator[](i);
  }

  void Shuffle();

 private:
  const T* matrix_;
  const T* targets_;
  const T* weights_;
  std::vector<size_t> sample_indices_;
};
}  // namespace internal
}  // namespace sapien
#endif  // INTERNAL_SAPIEN_UTILITY_SEQUENTIAL_DATASET_H_
