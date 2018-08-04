// Copyright 2018.

#include <random>
#include <algorithm>

#include "sapien/utility/seq_dataset.h"

namespace sapien {
namespace internal {

template<typename T>
SeqDataset<T>::SeqDataset(const size_t m, const size_t n,
                          const T* matrix, const T* targets, const T* weights)
    : n_features(n),
      n_samples(m),
      matrix_(matrix),
      targets_(targets),
      weights_(weights),
      sample_indices_(m) {
  for (size_t i = 0; i < m; ++i) {
    sample_indices_[i] = i;
  }
}

template<typename T>
SeqDataset<T>::~SeqDataset() {
  matrix_ = NULL;
  targets_ = NULL;
  weights_ = NULL;
}

template<typename T>
SeqDataset<T>::Sample::Sample() : x(NULL), target(0), weight(0) {
}

template<typename T>
SeqDataset<T>::Sample::Sample(const T* x, const T target, const T weight)
    : x(x), target(target), weight(weight) {
}

template<typename T>
SeqDataset<T>::Sample::Sample(const Sample& that)
    : x(that.x),
      target(that.target),
      weight(that.weight) {
}

template<typename T>
typename SeqDataset<T>::Sample&
SeqDataset<T>::Sample::operator=(const Sample& that) {
  if (this != &that) {
    x = that.x;
    target = that.target;
    weight = that.weight;
  }
  return *this;
}

template<typename T>
const typename SeqDataset<T>::Sample
SeqDataset<T>::operator[](const size_t i) const {
  const size_t sample_index = sample_indices_[i];
  T w = (weights_ == NULL) ? 1.0 : weights_[sample_index];
  return Sample(matrix_ + sample_index * n_features,
                targets_[sample_index], w);
}

// Randomly shuffle the sample_indices_ vector
template<typename T>
void
SeqDataset<T>::Shuffle() {
  std::random_device rd;
  std::mt19937 engine(rd());
  std::shuffle(sample_indices_.begin(), sample_indices_.end(), engine);
}

template class SeqDataset<float>;
template class SeqDataset<double>;
}  // namespace internal
}  // namespace sapien
