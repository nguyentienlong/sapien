// Copyright 2018.
//
// Author: mail2ngoclinh@gmail.com

#ifndef INCLUDE_SAPIEN_DATASET_H_
#define INCLUDE_SAPIEN_DATASET_H_

#include "sapien/internal/port.h"

namespace sapien {

// Load the MNIST dataset from a given root directory.
//
// X_train is an unsigned char array of size 60000 * 28 * 28.
// y_train is an unsigned char array of size 60000.
// X_test is an unsigned char array of size 10000 * 28 * 28.
// y_test is an unsigned char array of size 10000.
SAPIEN_EXPORT void LoadMNIST(const char* mnist_root_dir,
                             uint8_t* X_train,
                             uint8_t* y_train,
                             uint8_t* X_test,
                             uint8_t* y_test);
}  // namespace sapien
#endif  // INCLUDE_SAPIEN_DATASET_H_
