// Copyright 2018
//
// Author: mail2ngoclinh@gmail.com
//
// An example of using SGDClassifier to classify MNIST dataset.

#include <iostream>

#include "sapien/dataset.h"
#include "sapien/sgd/sgd_classifier.h"
#include "sapien/sgd/loss.h"
#include "glog/logging.h"

using sapien::uint8_t;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  if (argc != 2) {
    std::cout << "Usage: ./sgd_classifier_mnist mnist_root_dir" << std::endl;
    return -1;
  }

  const char* mnist_root_dir = argv[1];

  size_t n_train_images = 60000;
  size_t n_test_images = 10000;
  size_t n_features = 28*28;

  uint8_t* X_train_u8;
  uint8_t* X_test_u8;
  uint8_t* y_train_u8;
  uint8_t* y_test_u8;

  X_train_u8 = new uint8_t[n_train_images * n_features];
  y_train_u8 = new uint8_t[n_train_images];
  X_test_u8 = new uint8_t[n_test_images * n_features];
  y_test_u8 = new uint8_t[n_train_images];

  sapien::LoadMNIST(mnist_root_dir, X_train_u8, y_train_u8,
                    X_test_u8, y_test_u8);

  // First 10 train labels.
  // for (size_t i = 0; i < 10; ++i) {
  //   std::cout << static_cast<int>(y_train_u8[i]) << std::endl;
  // }

  // Convert data to double for training.

  double* X_train = new double[n_train_images * n_features];
  double* X_test = new double[n_test_images * n_features];

  // Populate X_train
  for (size_t i = 0; i < n_train_images * n_features; ++i) {
    X_train[i] = static_cast<double>(X_train_u8[i]);
  }

  // Populate X_test
  for (size_t i = 0; i < n_test_images * n_features; ++i) {
    X_test[i] = static_cast<double>(X_test_u8[i]);
  }

  // First train image
  // for (size_t i = 0; i < 28; ++i) {
  //   for (size_t j = 0; j < 28; ++j) {
  //     std::cout << X_train[i * 28 + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // Initialize the model

  sapien::sgd::SGDClassifier<uint8_t>::Options options;
  options.learning_rate_type = sapien::sgd::LEARNING_RATE_OPTIMAL;
  options.initial_learning_rate = 0.0;
  options.penalty_type = sapien::sgd::L2_PENALTY;
  options.penalty_strength = 1e-2;
  options.l1_ratio = 0.15;
  options.max_iter = 50;
  options.fit_intercept = true;
  options.average_sgd = 1;
  options.logging_type = sapien::sgd::SILENT;
  options.shuffle = true;

  sapien::sgd::SGDClassifier<uint8_t> model(options);
  model.loss_functor(new sapien::sgd::HingeLoss<double>(1.0));

  // Training

  model.Train(n_train_images, n_features, X_train, y_train_u8);
  double test_score = model.Score(n_test_images, n_features,
                                  X_test, y_test_u8);
  std::cout << model.Summary() << std::endl;
  std::cout << "Test score: " << test_score << std::endl;

  delete[] X_train_u8;
  delete[] y_train_u8;
  delete[] X_test_u8;
  delete[] y_test_u8;

  return 0;
}
