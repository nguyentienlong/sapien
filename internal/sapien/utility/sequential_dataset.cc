// Copyright 2018.

#include <random>
#include <algorithm>

#include "sapien/utility/sequential_dataset.h"

namespace sapien {
namespace internal {

template<typename T>
SequentialDataset<T>::SequentialDataset(const size_t m, const size_t n,
                                        const T* matrix, const T* targets,
                                        const T* weights)
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
SequentialDataset<T>::~SequentialDataset() {
  matrix_ = NULL;
  targets_ = NULL;
  weights_ = NULL;
}

template<typename T>
SequentialDataset<T>::Sample::Sample(const T* x, const T target, const T weight)
    : x(x), target(target), weight(weight) {
}

template<typename T>
const typename SequentialDataset<T>::Sample
SequentialDataset<T>::operator[](const size_t i) const {
  const size_t sample_index = sample_indices_[i];
  T w = (weights_ == NULL) ? 1.0 : weights_[sample_index];
  return Sample(matrix_ + sample_index * n_features,
                targets_[sample_index], w);
}

// Randomly shuffle the sample_indices_ vector
template<typename T>
void
SequentialDataset<T>::Shuffle() {
  std::random_device rd;
  std::mt19937 engine(rd());
  std::shuffle(sample_indices_.begin(), sample_indices_.end(), engine);
}

template class SequentialDataset<float>;
template class SequentialDataset<double>;
}  // namespace internal
}  // namespace sapien
