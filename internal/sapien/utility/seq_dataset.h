// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#ifndef INTERNAL_SAPIEN_UTILITY_SEQ_DATASET_H_
#define INTERNAL_SAPIEN_UTILITY_SEQ_DATASET_H_

#include <stddef.h>
#include <vector>

namespace sapien {
namespace internal {

template<typename T>
class SeqDataset {
 public:
  SeqDataset(const size_t m, const size_t n,
             const T* matrix, const T* targets, const T* weights = NULL);
  ~SeqDataset();

  const size_t n_features;
  const size_t n_samples;

  struct Sample {
    const T* x;
    T target;
    T weight;

    Sample();
    Sample(const T* x, const T target, const T weight);
    Sample(const Sample& that);
    Sample& operator=(const Sample& that);
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

  // We explicitly delete default constructor, copy constructor, and
  // assignment operator
  SeqDataset();
  SeqDataset(const SeqDataset&);
  SeqDataset& operator=(const SeqDataset&);
};
}  // namespace internal
}  // namespace sapien
#endif  // INTERNAL_SAPIEN_UTILITY_SEQ_DATASET_H_
